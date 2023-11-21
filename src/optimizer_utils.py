"""
Created by Andrew Silva on 11/15/23
Copied from https://git.ecker.tech/mrq/ai-voice-cloning and https://github.com/neonbjb/DL-Art-School

"""
import random
from collections import Counter, defaultdict
import math
from torch import distributed
from torch._C._distributed_c10d import ReduceOp
from torch.optim.lr_scheduler import LRScheduler

from utils import cfg_get, clip_grad_norm
from forward_pass_utils import get_injector
from loss_utils import create_loss, LossAccumulator
import logging
from collections import OrderedDict

import torch
from torch.cuda.amp import GradScaler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import Module

logger = logging.getLogger('base')


def recursively_detach(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    elif isinstance(v, list) or isinstance(v, tuple):
        out = [recursively_detach(i) for i in v]
        if isinstance(v, tuple):
            return tuple(out)
        return out
    elif isinstance(v, dict):
        out = {}
        for k, t in v.items():
            out[k] = recursively_detach(t)
        return out


def create_batch_size_optimizer(opt_train):
    if 'batch_size_optimizer' in opt_train.keys():
        if opt_train['batch_size_optimizer']['type'] == 'gradient_direction':
            return GradientDirectionOptimizer(opt_train)
    return MegabatchBatchSizeOptimizer(opt_train)


def grad(p):
    if p.grad is None:
        return torch.tensor(0)
    return p.grad.detach().clone()


# Base class for BatchSizeOptimizers.
class BatchSizeOptimizer:
    def focus(self, optimizer):
        pass

    def should_step(self, it):
        raise NotImplementedError

    def get_statistics(self):
        return {}


# BatchSizeOptimizer that just steps every megabatch.
class MegabatchBatchSizeOptimizer(BatchSizeOptimizer):
    def __init__(self, opt_train):
        pass

    def should_step(self, it):
        return True


# BatchSizeOptimizer that uses the gradient direction of a few parameters to determine when to step.
# Very similar to what is described in https://aclanthology.org/2020.acl-main.323.pdf
# Special note: this class will ALWAYS accumulate, at a minimum, 3 batches. Plan accordingly.
class GradientDirectionOptimizer(BatchSizeOptimizer):
    def __init__(self, training_cfg):
        self.opt = training_cfg['batch_size_optimizer']
        self.max_full_batches = cfg_get(self.opt, ['max_full_batches'], 10)
        self.parameters_to_poll = cfg_get(self.opt, ['poll_parameters'], 8)
        self.recalculate_directions_every = cfg_get(
            self.opt, ['recalculate_directions_steps'], 1)
        self.current_model = None

        # Metrics
        self.steps_taken = 0
        self.last_number_iterations = torch.zeros((128,))
        self.last_number_iterations_i = 0
        self.last_number_iterations_filled = False

    def vector_angle(self, v1, v2):
        if torch.all(v1 == 0) or torch.all(v2 == 0):
            return torch.tensor(0, device=v1.device)
        with torch.no_grad():
            v1 = v1.flatten()
            v2 = v2.flatten()
            v1_norm = (v1 ** 2).sum().sqrt()
            v2_norm = (v2 ** 2).sum().sqrt()
            angle = torch.arccos((torch.dot(v1, v2)) / (v1_norm * v2_norm))
            return angle

    def focus(self, model):
        if not hasattr(model, '_gradient_direction_optimizer_finished') or model._gradient_direction_optimizer_finished:
            all_params = list(filter(lambda t: '.weight' in t[0] and not hasattr(t[1].requires_grad, 'DO_NOT_TRAIN'),
                                     list(model.named_parameters())))  # Extracts weight parameters. Who cares about biases anyways? :)
            num_params = min(len(all_params), self.parameters_to_poll)
            model._gradient_direction_optimizer_params = random.sample(
                all_params, num_params)
            model._gradient_direction_optimizer_prior_directions = [
                0 for _ in range(num_params)]
            model._gradient_direction_optimizer_stopped_decreasing = [
                False for _ in range(num_params)]
            model._gradient_direction_optimizer_prior_grads = None
            model._gradient_direction_optimizer_step = 0
            model._gradient_direction_optimizer_finished = False
        self.current_model = model

    def should_step(self, it):
        model = self.current_model
        model._gradient_direction_optimizer_step += 1
        cur_grads = [grad(p)
                     for k, p in model._gradient_direction_optimizer_params]
        for cg in cur_grads:
            if torch.any(torch.isnan(cg)):
                print("BSO: found NaN. Passing it off to the GradScaler..")
                return True
        if model._gradient_direction_optimizer_prior_grads is not None:
            cur_dir = [self.vector_angle(lgrad, cgrad) for lgrad, cgrad in zip(
                model._gradient_direction_optimizer_prior_grads, cur_grads)]
            delta_dir = [(cdir - ldir) for cdir, ldir in zip(cur_dir,
                                                             model._gradient_direction_optimizer_prior_directions)]
            model._gradient_direction_optimizer_prior_directions = cur_dir
            model._gradient_direction_optimizer_stopped_decreasing = [sd or dd < 0 for sd, dd in zip(
                model._gradient_direction_optimizer_stopped_decreasing, delta_dir)]
            all_finished = all(
                model._gradient_direction_optimizer_stopped_decreasing)

            # For distributed optimizers, like ZeroRedundancyAdam, we need to reach a consensus as to whether or not to reduce.
            if distributed.is_initialized() and distributed.get_world_size() > 1:
                all_finished = torch.tensor(all_finished)
                distributed.all_reduce(all_finished, ReduceOp.BAND)
                all_finished = torch.all(all_finished)

            if all_finished or model._gradient_direction_optimizer_step >= self.max_full_batches:
                # <0 means the gradient direction is getting larger. Halt batch accumulation here.
                model._gradient_direction_optimizer_finished = True
                self.record_number_steps(
                    model._gradient_direction_optimizer_step)
                # Fix the gradients. We've accumulated _gradient_direction_optimizer_step steps total, so we need to divide the grads by that.
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad = p.grad / model._gradient_direction_optimizer_step
                return True
        model._gradient_direction_optimizer_prior_grads = cur_grads
        return False

    def record_number_steps(self, steps):
        self.last_number_iterations[self.last_number_iterations_i] = steps
        if self.last_number_iterations_i == self.last_number_iterations.shape[0]-1:
            self.last_number_iterations_filled = True
        self.last_number_iterations_i = (
            self.last_number_iterations_i + 1) % self.last_number_iterations.shape[0]
        self.steps_taken += 1

    def get_statistics(self):
        res = {"batch_size_opt_total_steps": self.steps_taken}
        if self.last_number_iterations_filled:
            res["batch_size_opt_avg_iterations_per_step"] = self.last_number_iterations.mean(
            ).item()
        else:
            res["batch_size_opt_avg_iterations_per_step"] = self.last_number_iterations[:
                                                                                        self.last_number_iterations_i].mean().item()
        return res


# Defines the expected API for a single training step
class ConfigurableStep(Module):

    def __init__(self, step_cfg, env):
        super(ConfigurableStep, self).__init__()

        self.step_cfg = step_cfg
        self.env = env
        self.cfg = env['cfg']
        self.gen_outputs = step_cfg['generator_outputs']
        self.loss_accumulator = LossAccumulator(
            buffer_sz=cfg_get(step_cfg, ['loss_log_buffer'], 50))
        self.optimizers = None
        self.scaler = GradScaler(enabled=self.cfg['fp16'] or cfg_get(
            self.cfg, ['grad_scaler_enabled'], False))
        self.grads_generated = False
        self.clip_grad_eps = cfg_get(step_cfg, ['clip_grad_eps'], None)

        # This is a half-measure that can be used between anomaly_detection and running a potentially problematic
        # trainer bare. With this turned on, the optimizer will not step() if a nan grad is detected. If a model trips
        # this warning 10 times in a row, the training session is aborted and the model state is saved. This has a
        # noticeable affect on training speed, but nowhere near as bad as anomaly_detection.
        self.check_grads_for_nan = cfg_get(
            step_cfg, ['check_grads_for_nan'], False)
        self.nan_counter = 0
        # This is a similar mechanism plugged into the forward() pass. It cannot be turned off.
        self.nan_loss_counter = 0

        self.injectors = []
        if 'injectors' in self.step_cfg.keys():
            injector_names = []
            for inj_name, injector in self.step_cfg['injectors'].items():
                # Repeated names are always an error case.
                assert inj_name not in injector_names
                injector_names.append(inj_name)
                self.injectors.append(get_injector(injector, env))

        losses = []
        self.weights = {}
        if 'losses' in self.step_cfg.keys():
            for loss_name, loss in self.step_cfg['losses'].items():
                # Repeated names are always an error case.
                assert loss_name not in self.weights.keys()
                losses.append((loss_name, create_loss(loss, env)))
                self.weights[loss_name] = loss['weight']
        self.losses = OrderedDict(losses)

    def get_network_for_name(self, name):
        return self.env['generators'][name] if name in self.env['generators'].keys() \
            else self.env['discriminators'][name]

    # Subclasses should override this to define individual optimizers. They should all go into self.optimizers.
    #  This default implementation defines a single optimizer for all Generator parameters.
    #  Must be called after networks are initialized and wrapped.
    def define_optimizers(self):
        opt_configs = [cfg_get(self.step_cfg, ['optimizer_params'], None)]
        self.optimizers = []
        if opt_configs[0] is None:
            return
        training = self.step_cfg['training']
        training_net = self.get_network_for_name(training)
        nets = [training_net]
        training = [training]
        for net_name, net, opt_config in zip(training, nets, opt_configs):
            # Configs can organize parameters by-group and specify different learning rates for each group. This only
            # works in the model specifically annotates which parameters belong in which group using PARAM_GROUP.
            optim_params = {'default': {'params': [], 'lr': opt_config['lr']}}
            if opt_config is not None and 'param_groups' in opt_config.keys():
                for k, pg in opt_config['param_groups'].items():
                    optim_params[k] = {'params': [], 'lr': pg['lr']}

            import torch.nn as nn
            norm_modules = (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d, nn.InstanceNorm1d,
                            nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm, nn.LayerNorm)
            # nn.Embedding
            emb_modules = (nn.Embedding, nn.EmbeddingBag)
            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for mn, m in net.named_modules():
                for k, v in m.named_parameters():
                    v.is_bias = k.endswith(".bias")
                    v.is_weight = k.endswith(".weight")
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)

                    fpn = '%s.%s' % (mn, k) if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

                    # Some models can specify some parameters to be in different groups.
                    param_group = "default"
                    if hasattr(v, 'PARAM_GROUP'):
                        if v.PARAM_GROUP in optim_params.keys():
                            param_group = v.PARAM_GROUP
                        else:
                            logger.warning(f'Model specifies a custom param group {v.PARAM_GROUP} which is not configured. '
                                           f'The same LR will be used for all parameters.')

                    if v.requires_grad:
                        optim_params[param_group]['params'].append(v)
                    else:
                        if self.env['rank'] <= 0:
                            logger.warning(
                                'Params [{:s}] will not optimize.'.format(k))
            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(
                list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]

            if 'optimizer' not in self.step_cfg.keys() or self.step_cfg['optimizer'] == 'adamw':
                groups = [
                    {'params': params_weights, 'weight_decay': cfg_get(
                        opt_config, ['weight_decay'], 0)},
                    {'params': params_notweights, 'weight_decay': 0}
                ]
                # torch.optim.AdamW
                opt = torch.optim.AdamW(groups, lr=opt_config['lr'],
                                        weight_decay=cfg_get(
                                   opt_config, ['weight_decay'], 1e-2),
                               betas=(cfg_get(opt_config, ['beta1'], .9), cfg_get(opt_config, ['beta2'], .999)))
                opt._group_names = [
                    params_names_weights, params_names_notweights]
            elif self.step_cfg['optimizer'] == 'mu_adamw':
                groups = [
                    {'params': params_weights, 'weight_decay': cfg_get(
                        opt_config, ['weight_decay'], 0)},
                    {'params': params_notweights, 'weight_decay': 0}
                ]
                from mup.optim import MuAdamW
                opt = MuAdamW(groups, lr=opt_config['lr'],
                              weight_decay=cfg_get(
                                  opt_config, ['weight_decay'], 1e-2),
                              betas=(cfg_get(opt_config, ['beta1'], .9), cfg_get(opt_config, ['beta2'], .999)))
                opt._group_names = [
                    params_names_weights, params_names_notweights]
            elif self.step_cfg['optimizer'] == 'adamw_zero':
                # The torch ZeRO implementation does not seem to support parameter groups, so do not shard the non-weighted
                # parameters and just use a normal AdamW implementation. In a large network, these weights will normally
                # be a tiny fraction of the total weights.
                # torch.optim.AdamW
                opt_unweighted = torch.optim.AdamW(params_notweights, lr=opt_config['lr'], weight_decay=0,
                                          betas=(cfg_get(opt_config, ['beta1'], .9), cfg_get(opt_config, ['beta2'], .999)))
                opt_unweighted._config = opt_config
                opt_unweighted._config['network'] = net_name
                opt_unweighted._group_names = []
                self.optimizers.append(opt_unweighted)

                # torch.optim.AdamW
                opt = ZeroRedundancyOptimizer(params_weights, optimizer_class=torch.optim.AdamW, lr=opt_config['lr'],
                                              weight_decay=cfg_get(
                                                  opt_config, ['weight_decay'], 1e-2),
                                              betas=(cfg_get(opt_config, ['beta1'], .9), cfg_get(opt_config, ['beta2'], .999)))
                opt.param_groups[0]['initial_lr'] = opt_config['lr']
                opt._group_names = []
            elif self.step_cfg['optimizer'] == 'sgd':
                from torch.optim import SGD
                opt = SGD(list(optim_params.values(
                )), lr=opt_config['lr'], momentum=opt_config['momentum'], weight_decay=opt_config['weight_decay'])
                opt._group_names = sorted(list(all_param_names))
            # This is a bit seedy, but we will need these configs later.
            opt._config = opt_config
            opt._config['network'] = net_name
            self.optimizers.append(opt)

    # Returns all optimizers used in this step.
    def get_optimizers(self):
        assert self.optimizers is not None
        return self.optimizers

    # Returns optimizers which are opting in for default LR scheduling.
    def get_optimizers_with_default_scheduler(self):
        assert self.optimizers is not None
        return self.optimizers

    # Returns the names of the networks this step will train. Other networks will be frozen.
    def get_networks_trained(self):
        if isinstance(self.step_cfg['training'], list):
            return self.step_cfg['training']
        else:
            return [self.step_cfg['training']]

    def get_training_network_name(self):
        if isinstance(self.step_cfg['training'], list):
            return self.step_cfg['training'][0]
        else:
            return self.step_cfg['training']

    # Performs all forward and backward passes for this step given an input state. All input states are lists of
    # chunked tensors. Use grad_accum_step to dereference these steps. Should return a dict of tensors that later
    # steps might use. These tensors are automatically detached and accumulated into chunks.
    def do_forward_backward(self, state, grad_accum_step, amp_loss_id, train=True, no_ddp_sync=False, loss_accumulator=None):
        # <-- Will store the entire local state to be passed to injectors & losses.
        local_state = {}
        # <-- Will store state values created by this step for returning to ExtensibleTrainer.
        new_state = {}
        for k, v in state.items():
            local_state[k] = v[grad_accum_step]
        local_state['train_nets'] = str(self.get_networks_trained())
        loss_accumulator = self.loss_accumulator if loss_accumulator is None else loss_accumulator

        # Some losses compute backward() internally. Accommodate this by stashing the amp_loss_id in env.
        self.env['amp_loss_id'] = amp_loss_id
        self.env['current_step_optimizers'] = self.optimizers
        self.env['training'] = train

        # Inject in any extra dependencies.
        for inj in self.injectors:
            # Don't do injections tagged with eval unless we are not in train mode.
            if train and 'eval' in inj.cfg.keys() and inj.cfg['eval']:
                continue
            # Likewise, don't do injections tagged with train unless we are not in eval.
            if not train and 'train' in inj.cfg.keys() and inj.cfg['train']:
                continue
            # Don't do injections tagged with 'after' or 'before' when we are out of spec.
            if 'after' in inj.cfg.keys() and self.env['step'] < inj.cfg['after'] or \
               'before' in inj.cfg.keys() and self.env['step'] > inj.cfg['before'] or \
               'every' in inj.cfg.keys() and self.env['step'] % inj.cfg['every'] != 0:
                continue
            if 'no_accum' in inj.cfg.keys() and grad_accum_step > 0:
                continue
            training_net = self.get_network_for_name(self.step_cfg['training'])
            if no_ddp_sync and hasattr(training_net, 'no_sync'):
                with training_net.no_sync():
                    injected = inj(local_state)
            elif cfg_get(inj.cfg, ['no_grad'], False):
                with torch.no_grad():
                    injected = inj(local_state)
            else:
                injected = inj(local_state)
            local_state.update(injected)
            new_state.update(injected)

            if hasattr(inj, 'extra_metrics'):
                for n, v in inj.extra_metrics().items():
                    # Doesn't really work for training setups where multiple of the same injector are used.
                    loss_accumulator.add_loss(n, v)

        if len(self.losses) > 0:
            # Finally, compute the losses.
            total_loss = 0
            for loss_name, loss in self.losses.items():
                multiplier = 1
                # Some losses only activate after a set number of steps. For example, proto-discriminator losses can
                # be very disruptive to a generator.
                if 'after' in loss.cfg.keys() and loss.cfg['after'] > self.env['step'] or \
                   'before' in loss.cfg.keys() and self.env['step'] > loss.cfg['before'] or \
                   'every' in loss.cfg.keys() and self.env['step'] % loss.cfg['every'] != 0:
                    # Multiply by 0 so gradients still flow and DDP works. Effectively this means the loss is unused.
                    multiplier = 0

                l = loss(self.get_network_for_name(
                    self.step_cfg['training']), local_state)
                if not l.isfinite():
                    print(f'!!Detected non-finite loss {loss_name}')
                total_loss += l * self.weights[loss_name] * multiplier
                # Record metrics.
                if isinstance(l, torch.Tensor):
                    loss_accumulator.add_loss(loss_name, l)

            # In some cases, the loss could not be set (e.g. all losses have 'after')
            if train and isinstance(total_loss, torch.Tensor) and total_loss.isfinite():
                loss_accumulator.add_loss("%s_total" % (
                    self.get_training_network_name(),), total_loss)

                # Scale the loss down by the accumulation factor.
                total_loss = total_loss / self.env['mega_batch_factor']

                # Get dem grads!
                self.scaler.scale(total_loss).backward()
                self.grads_generated = True
                # Reset nan_loss_counter
                self.nan_loss_counter = 0
            elif not total_loss.isfinite():
                print("Non-finite loss encountered. Skipping backwards step.")
                self.nan_loss_counter += 1
                if self.nan_loss_counter > 10:
                    print(
                        "Encountered 10 NaN losses in a row. Something is screwed up. Dumping model weights and exiting.")
                    if self.env['rank'] == 0:
                        torch.save(training_net.state_dict(),
                                   "nan_error_weights.pth")
                    exit(1)

        # Detach all state variables. Within the step, gradients can flow. Once these variables leave the step
        # we must release the gradients.
        new_state = recursively_detach(new_state)

        # Prune state outputs that are not actually needed.
        if 'step_outputs' in self.step_cfg.keys():
            nst = {}
            for k in self.step_cfg['step_outputs']:
                nst[k] = new_state[k]
            new_state = nst

        return new_state

    # Performs the optimizer step after all gradient accumulation is completed. Default implementation simply steps()
    # all self.optimizers.
    def do_step(self, step):
        if not self.grads_generated:
            return
        self.grads_generated = False
        for opt in self.optimizers:
            # self.scaler.unscale_(opt) It would be important to do this here, but ExtensibleTrainer currently does it.

            # Optimizers can be opted out in the early stages of training.
            after = opt._config['after'] if 'after' in opt._config.keys(
            ) else 0
            after_network = self.cfg['networks'][opt._config['network']
                                                 ]['after'] if 'after' in self.cfg['networks'][opt._config['network']].keys() else 0
            after = max(after, after_network)
            if self.env['step'] < after:
                continue
            before = opt._config['before'] if 'before' in opt._config.keys(
            ) else -1
            if before != -1 and self.env['step'] > before:
                continue

            nan_found = False
            if self.check_grads_for_nan:
                for pg in opt.param_groups:
                    for p in pg['params']:
                        if not torch.isfinite(p.grad).any():
                            nan_found = True
                            break
                    if nan_found:
                        break
                if nan_found:
                    print("NaN found in grads. Throwing this step out.")
                    self.nan_counter += 1
                else:
                    self.nan_counter = 0

            if self.clip_grad_eps is not None and self.clip_grad_eps != 0:
                for pgn, pg in zip(opt._group_names, opt.param_groups):
                    grad_norm = clip_grad_norm(
                        pg['params'], self.clip_grad_eps)
                    if torch.isnan(grad_norm):
                        print(
                            "NaN found in clip_grad; zeroing grad and trying again.")
                        nan_found = True
                        self.nan_counter += 1

            if not nan_found:
                self.scaler.step(opt)
                self.scaler.update()
            else:
                opt.zero_grad()

    def get_metrics(self):
        metrics = self.loss_accumulator.as_dict()
        metrics['grad_scaler_scale'] = self.scaler.get_scale()
        return metrics


def get_scheduler_for_name(name, optimizers, scheduler_opt):
    schedulers = []
    for o in optimizers:
        # Hack to support LARC, which wraps an underlying optimizer.
        if hasattr(o, 'optim'):
            o = o.optim

        if name == 'MultiStepLR':
            sched = MultiStepLR_Restart(o, scheduler_opt['gen_lr_steps'],
                                        restarts=scheduler_opt['restarts'],
                                        weights=scheduler_opt['restart_weights'],
                                        gamma=scheduler_opt['lr_gamma'],
                                        clear_state=scheduler_opt['clear_state'],
                                        force_lr=scheduler_opt['force_lr'],
                                        warmup_steps=cfg_get(scheduler_opt, ['warmup_steps'], 0))
        elif name == 'ProgressiveMultiStepLR':
            sched = ProgressiveMultiStepLR(o, scheduler_opt['gen_lr_steps'],
                                           scheduler_opt['progressive_starts'],
                                           scheduler_opt['lr_gamma'])
        elif name == 'CosineAnnealingLR_Restart':
            sched = CosineAnnealingLR_Restart(
                o, scheduler_opt['T_period'], scheduler_opt['warmup'], eta_min=scheduler_opt['eta_min'],
                restarts=scheduler_opt['restarts'], weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError('Scheduler not available')
        schedulers.append(sched)
    return schedulers


# This scheduler is specifically designed to modulate the learning rate of several different param groups configured
# by a generator or discriminator that slowly adds new stages one at a time, e.g. like progressive growing of GANs.
class ProgressiveMultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, group_starts, gamma=0.1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.group_starts = group_starts
        super(ProgressiveMultiStepLR, self).__init__(optimizer)

    def get_lr(self):
        group_lrs = []
        assert len(self.optimizer.param_groups) == len(self.group_starts)
        for group, group_start in zip(self.optimizer.param_groups, self.group_starts):
            if self.last_epoch - group_start not in self.milestones:
                group_lrs.append(group['lr'])
            else:
                group_lrs.append(group['lr'] * self.gamma)
        return group_lrs


class MultiStepLR_Restart(LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, force_lr=False, last_epoch=-1, warmup_steps=0):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.force_lr = force_lr
        if force_lr:
            print(f"!!Forcing the learning rate to: {force_lr}")
        self.warmup_steps = warmup_steps
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Note to self: for the purposes of this trainer, "last_epoch" should read "last_step"
        if self.force_lr is not None:
            return [self.force_lr for _ in self.optimizer.param_groups]
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            factor = 1 - (self.warmup_steps - self.last_epoch) / \
                self.warmup_steps
            return [group['initial_lr'] * factor for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    # Allow this scheduler to use newly appointed milestones partially through a training run..
    def load_state_dict(self, s):
        milestones_cache = self.milestones
        force_lr_cache = self.force_lr
        super(MultiStepLR_Restart, self).load_state_dict(s)
        self.milestones = milestones_cache
        self.force_lr = force_lr_cache


class CosineAnnealingLR_Restart(LRScheduler):
    def __init__(self, optimizer, T_period, warmup=0, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.warmup = warmup
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch - self.warmup
        if step <= 0:
            return self.base_lrs
        elif step in self.restarts:
            self.last_restart = step
            self.T_max = self.T_period[self.restarts.index(step) + 1]
            weight = self.restart_weights[self.restarts.index(step)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (step - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (step - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((step - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]