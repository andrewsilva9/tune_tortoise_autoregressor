"""
Created by Andrew Silva on 11/13/23
Copied from https://git.ecker.tech/mrq/ai-voice-cloning and https://github.com/neonbjb/DL-Art-School

"""
import copy
import json
import logging
import math
import os
import shutil

from time import time
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn

from utils import (cfg_get, mkdirs, load_network, mkdir_and_rename,
                   setup_logger, dict2str, dict_to_nonedict, set_random_seed)

from optimizer_utils import create_batch_size_optimizer, ConfigurableStep, get_scheduler_for_name, LossAccumulator
from forward_pass_utils import get_injector
from unified_voice_network import get_model
from dataset import get_dataset, get_dataset_debugger, create_dataloader

logger = logging.getLogger('base')


def try_json(data):
    reduced = {}
    for k, v in data.items():
        try:
            json.dumps(v)
        except Exception as e:
            continue
        reduced[k] = v
    return json.dumps(reduced)


def process_metrics(metrics):
    reduced = {}
    for metric in metrics:
        d = metric.as_dict() if hasattr(metric, 'as_dict') else metric
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and len(v.shape) == 0:
                if k in reduced.keys():
                    reduced[k].append(v)
                else:
                    reduced[k] = [v]
    logs = {}

    for k, v in reduced.items():
        logs[k] = torch.stack(v).mean().item()

    return logs


class ExtensibleTrainer:
    def __init__(self, cfg):
        self.rank = -1  # non dist training
        self.cfg = cfg
        train_opt = cfg['train']
        self.schedulers = []
        self.optimizers = []
        self.is_train = cfg['is_train']
        self.device = torch.cuda.current_device() if cfg['gpu_ids'] else torch.device('cpu')
        # env is used as a global state to store things that subcomponents might need.
        self.env = {'device': self.device,
                    'rank': self.rank,
                    'cfg': cfg,
                    'step': 0,
                    'dist': cfg['dist']
                    }
        if cfg['path']['models'] is not None:
            self.env['base_path'] = os.path.join(cfg['path']['models'])

        self.mega_batch_factor = 1
        if self.is_train:
            self.mega_batch_factor = train_opt['mega_batch_factor']
            self.env['mega_batch_factor'] = self.mega_batch_factor
            self.batch_factor = self.mega_batch_factor
            self.ema_rate = cfg_get(train_opt, ['ema_rate'], .999)
            # It is advantageous for large networks to do this to save an extra copy of the model weights.
            # It does come at the cost of a round trip to CPU memory at every batch.
            self.do_emas = cfg_get(train_opt, ['ema_enabled'], True)
            self.ema_on_cpu = cfg_get(train_opt, ['ema_on_cpu'], False)
        self.checkpointing_cache = cfg['checkpointing_enabled']
        self.auto_recover = cfg_get(
            cfg, ['automatically_recover_nan_by_reverting_n_saves'], None)
        self.batch_size_optimizer = create_batch_size_optimizer(train_opt)
        self.auto_scale_grads = cfg_get(
            cfg, ['automatically_scale_grads_for_fanin'], False)
        self.auto_scale_basis = cfg_get(
            cfg, ['automatically_scale_base_layer_size'], 1024)

        self.netsG = {}
        self.netsD = {}
        for name, net_cfg in cfg['networks'].items():
            # Trainable is a required parameter, but the default is simply true. Set it here.
            if 'trainable' not in net_cfg.keys():
                net_cfg['trainable'] = True
            new_net = get_model(net_cfg).to(self.device)
            if net_cfg['type'] == 'generator':
                self.netsG[name] = new_net
            elif net_cfg['type'] == 'discriminator':
                self.netsD[name] = new_net
            else:
                raise NotImplementedError(
                    "Can only handle generators and discriminators")

            if not net_cfg['trainable']:
                new_net.eval()
            if net_cfg['wandb_debug']:
                import wandb
                wandb.watch(new_net, log='all', log_freq=3)

        # Initialize the train/eval steps
        self.step_names = []
        self.steps = []
        for step_name, step in cfg['steps'].items():
            step = ConfigurableStep(step, self.env)
            # This could be an OrderedDict, but it's a PITA to integrate with AMP below.
            self.step_names.append(step_name)
            self.steps.append(step)

        # step.define_optimizers() relies on the networks being placed in the env, so put them there. Even though
        # they aren't wrapped yet.
        self.env['generators'] = self.netsG
        self.env['discriminators'] = self.netsD

        # Define the optimizers from the steps
        for s in self.steps:
            s.define_optimizers()
            self.optimizers.extend(s.get_optimizers())

        if self.is_train:
            # Find the optimizers that are using the default scheduler, then build them.
            def_opt = []
            for s in self.steps:
                def_opt.extend(s.get_optimizers_with_default_scheduler())
            self.schedulers = get_scheduler_for_name(train_opt['default_lr_scheme'], def_opt, train_opt)

            # Set the starting step count for the scheduler.
            for sched in self.schedulers:
                sched.last_epoch = cfg['current_step']
        else:
            self.schedulers = []

        self.emas = {}
        if self.do_emas:
            self.emas = copy.deepcopy(self.netsG)
        self.networks = self.netsG
        # Replace the env networks with the wrapped networks
        self.env['generators'] = self.netsG
        self.env['discriminators'] = self.netsD
        self.env['emas'] = self.emas

        # self.print_network()  # print network
        self.load()  # load networks from save states as needed

        # Setting this to false triggers SRGAN to call the models update_model() function on the first iteration.
        self.updated = True

    def feed_data(self, data, step, need_GT=True, perform_micro_batching=True):
        self.env['step'] = step
        self.batch_factor = self.mega_batch_factor
        self.cfg['checkpointing_enabled'] = self.checkpointing_cache
        # The batch factor can be adjusted on a period to allow known high-memory steps to fit in GPU memory.
        if 'train' in self.cfg.keys() and \
                'mod_batch_factor' in self.cfg['train'].keys() and \
                self.env['step'] % self.cfg['train']['mod_batch_factor_every'] == 0:
            self.batch_factor = self.cfg['train']['mod_batch_factor']
            if self.cfg['train']['mod_batch_factor_also_disable_checkpointing']:
                self.cfg['checkpointing_enabled'] = False

        self.eval_state = {}
        for o in self.optimizers:
            o.zero_grad()
        torch.cuda.empty_cache()

        sort_key = cfg_get(self.cfg, ['train', 'sort_key'], None)
        if sort_key is not None:
            sort_indices = torch.sort(data[sort_key], descending=True).indices
        else:
            sort_indices = None

        batch_factor = self.batch_factor if perform_micro_batching else 1
        self.dstate = {}
        for k, v in data.items():
            if sort_indices is not None:
                if isinstance(v, list):
                    v = [v[i] for i in sort_indices]
                else:
                    v = v[sort_indices]
            if isinstance(v, torch.Tensor):
                self.dstate[k] = [t.to(self.device) for t in torch.chunk(
                    v, chunks=batch_factor, dim=0)]

        if cfg_get(self.cfg, ['train', 'auto_collate'], False):
            for k, v in self.dstate.items():
                if f'{k}_lengths' in self.dstate.keys():
                    for c in range(len(v)):
                        maxlen = self.dstate[f'{k}_lengths'][c].max()
                        if len(v[c].shape) == 2:
                            self.dstate[k][c] = self.dstate[k][c][:, :maxlen]
                        elif len(v[c].shape) == 3:
                            self.dstate[k][c] = self.dstate[k][c][:,
                                                                  :, :maxlen]
                        elif len(v[c].shape) == 4:
                            self.dstate[k][c] = self.dstate[k][c][:,
                                                                  :, :, :maxlen]

    def optimize_parameters(self, it, optimize=True, return_grad_norms=False):
        grad_norms = {}

        # Some models need to make parametric adjustments per-step. Do that here.
        for net in self.networks.values():
            if hasattr(net, 'update_for_step'):
                net.update_for_step(it, os.path.join(self.cfg['path']['models'], "../.."))
            elif hasattr(net, 'module') and hasattr(net.module, "update_for_step"):
                net.module.update_for_step(it, os.path.join(
                    self.cfg['path']['models'], "../.."))

        # Iterate through the steps, performing them one at a time.
        state = self.dstate
        for step_num, step in enumerate(self.steps):
            train_step = True
            # 'every' is used to denote steps that should only occur at a certain integer factor rate. e.g. '2' occurs every 2 steps.
            # Note that the injection points for the step might still be required, so address this by setting train_step=False
            if 'every' in step.step_cfg.keys() and it % step.step_cfg['every'] != 0:
                train_step = False
            # Steps can opt out of early (or late) training, make sure that happens here.
            if 'after' in step.step_cfg.keys() and it < step.step_cfg['after'] or 'before' in step.step_cfg.keys() and it > step.step_cfg['before']:
                continue
            # Steps can choose to not execute if a state key is missing.
            if 'requires' in step.step_cfg.keys():
                requirements_met = True
                for requirement in step.step_cfg['requires']:
                    if requirement not in state.keys():
                        requirements_met = False
                if not requirements_met:
                    continue

            if train_step:
                # Only set requires_grad=True for the network being trained.
                nets_to_train = step.get_networks_trained()
                enabled = 0
                for name, net in self.networks.items():
                    net_enabled = name in nets_to_train
                    if net_enabled:
                        enabled += 1
                    # Networks can opt out of training before a certain iteration by declaring 'after' in their definition.
                    if 'after' in self.cfg['networks'][name].keys() and it < self.cfg['networks'][name]['after']:
                        net_enabled = False
                    for p in net.parameters():
                        do_not_train_flag = hasattr(p, "DO_NOT_TRAIN") or (
                            hasattr(p, "DO_NOT_TRAIN_UNTIL") and it < p.DO_NOT_TRAIN_UNTIL)
                        if p.dtype != torch.int64 and p.dtype != torch.bool and not do_not_train_flag:
                            p.requires_grad = net_enabled
                        else:
                            p.requires_grad = False
                assert enabled == len(nets_to_train)

                for o in step.get_optimizers():
                    o.zero_grad()

            # Now do a forward and backward pass for each gradient accumulation step.
            new_states = {}
            self.batch_size_optimizer.focus(net)
            for m in range(self.batch_factor):
                ns = step.do_forward_backward(
                    state, m, step_num, train=train_step, no_ddp_sync=(m+1 < self.batch_factor))
                # Call into post-backward hooks.
                for name, net in self.networks.items():
                    if hasattr(net, 'after_backward'):
                        net.after_backward(it)
                    elif hasattr(net, 'module') and hasattr(net.module, 'after_backward'):
                        net.module.after_backward(it)

                for k, v in ns.items():
                    if k not in new_states.keys():
                        new_states[k] = [v]
                    else:
                        new_states[k].append(v)

            # Push the detached new state tensors into the state map for use with the next step.
            for k, v in new_states.items():
                if k in state.keys():
                    raise ValueError(f'{k} already exists in keys: {list(state.keys())}')
                state[k] = v

            # (Maybe) perform a step.
            if train_step and optimize and self.batch_size_optimizer.should_step(it):
                # Unscale gradients within the step. (This is admittedly pretty messy but the API contract between step & ET is pretty much broken at this point)
                # This is needed to accurately log the grad norms.
                for opt in step.optimizers:
                    from torch.cuda.amp.grad_scaler import OptState
                    if step.scaler.is_enabled() and step.scaler._per_optimizer_states[id(opt)]["stage"] is not OptState.UNSCALED:
                        step.scaler.unscale_(opt)

                # Call into pre-step hooks.
                for name, net in self.networks.items():
                    if hasattr(net, 'before_step'):
                        net.before_step(it)
                    elif hasattr(net, 'module') and hasattr(net.module, 'before_step'):
                        net.module.before_step(it)

                if self.auto_scale_grads:
                    asb = math.sqrt(self.auto_scale_basis)
                    for net in self.networks.values():
                        for mod in net.modules():
                            fan_in = -1
                            if isinstance(mod, nn.Linear):
                                fan_in = mod.weight.data.shape[1]
                            elif isinstance(mod, nn.Conv1d):
                                fan_in = mod.weight.data.shape[0]
                            elif isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Conv3d):
                                assert "Not yet implemented!"
                            if fan_in != -1:
                                p = mod.weight
                                if hasattr(p, 'grad') and p.grad is not None:
                                    p.grad = p.grad * asb / math.sqrt(fan_in)

                if return_grad_norms and train_step:
                    for name in nets_to_train:
                        model = self.networks[name]
                        if hasattr(model, 'get_grad_norm_parameter_groups'):
                            pgroups = {
                                f'{name}_{k}': v for k, v in model.get_grad_norm_parameter_groups().items()}
                        elif hasattr(model, 'module') and hasattr(model.module, 'get_grad_norm_parameter_groups'):
                            pgroups = {
                                f'{name}_{k}': v for k, v in model.module.get_grad_norm_parameter_groups().items()}
                        else:
                            pgroups = {f'{name}_all_parameters': list(
                                model.parameters())}
                    for name in pgroups.keys():  # TODO: Bug? Should be indented?
                        stacked_grads = []
                        for p in pgroups[name]:
                            if hasattr(p, 'grad') and p.grad is not None:
                                stacked_grads.append(
                                    torch.norm(p.grad.detach(), 2))
                        if not stacked_grads:
                            continue
                        grad_norms[name] = torch.norm(
                            torch.stack(stacked_grads), 2)
                        grad_norms[name] = grad_norms[name].cpu()

                self.consume_gradients(state, step, it)

        return grad_norms

    def consume_gradients(self, state, step, it):
        step.do_step(it)

        # Call into custom step hooks as well as update EMA params.
        for name, net in self.networks.items():
            if hasattr(net, 'after_step'):
                net.after_step(it)
            elif hasattr(net, 'module') and hasattr(net.module, "after_step"):
                net.module.after_step(it)
            if self.do_emas:
                # When the EMA is on the CPU, only update every 10 steps to save processing time.
                if self.ema_on_cpu and it % 10 != 0:
                    continue
                ema_params = self.emas[name].parameters()
                net_params = net.parameters()
                for ep, np in zip(ema_params, net_params):
                    ema_rate = self.ema_rate
                    new_rate = 1 - ema_rate
                    if self.ema_on_cpu:
                        np = np.cpu()
                        # Because it only happens every 10 steps.
                        ema_rate = ema_rate ** 10
                        mid = (1 - (ema_rate+new_rate))/2
                        ema_rate += mid
                        new_rate += mid
                    ep.detach().mul_(ema_rate).add_(np, alpha=1 - ema_rate)

    def test(self):
        for net in self.netsG.values():
            net.eval()

        accum_metrics = LossAccumulator(buffer_sz=len(self.steps))
        with torch.no_grad():
            # This can happen one of two ways: Either a 'validation injector' is provided, in which case we run that.
            # Or, we run the entire chain of steps in "train" mode and use eval.output_state.
            if 'injectors' in self.cfg['eval'].keys():
                state = {}
                for inj_cfg in self.cfg['eval']['injectors'].values():
                    # Need to move from mega_batch mode to batch mode (remove chunks)
                    for k, v in self.dstate.items():
                        state[k] = v[0]
                    inj = get_injector(inj_cfg, self.env)
                    state.update(inj(state))
            else:
                # Iterate through the steps, performing them one at a time.
                state = self.dstate
                for step_num, s in enumerate(self.steps):
                    ns = s.do_forward_backward(
                        state, 0, step_num, train=False, loss_accumulator=accum_metrics)
                    for k, v in ns.items():
                        state[k] = [v]

        for net in self.netsG.values():
            net.train()
        return accum_metrics

    # Fetches a summary of the log.
    def get_current_log(self, step):
        log = {}
        for s in self.steps:
            log.update(s.get_metrics())

        # Some generators can do their own metric logging.
        for net_name, net in self.networks.items():
            if hasattr(net, 'get_debug_values'):
                log.update(net.get_debug_values(step, net_name))
            elif hasattr(net, 'module') and hasattr(net.module, "get_debug_values"):
                log.update(net.module.get_debug_values(step, net_name))

        # Log learning rate (from first param group) too.
        for o in self.optimizers:
            for pgi, pg in enumerate(o.param_groups):
                log['learning_rate_%s_%i' %
                    (o._config['network'], pgi)] = pg['lr']

        # The batch size optimizer also outputs loggable data.
        log.update(self.batch_size_optimizer.get_statistics())
        return log

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr']
                                    for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.last_epoch = cur_iter
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def load(self):
        for netdict in [self.netsG, self.netsD]:
            for name, net in netdict.items():
                load_path = self.cfg['path']['pretrain_model_%s' % (name,)]
                if load_path is None:
                    return
                logger.info('Loading model for [%s]' % (load_path,))
                load_network(load_path, net, self.cfg['path']['strict_load'], cfg_get(
                    self.cfg, ['path', f'pretrain_base_path_{name}']))
                load_path_ema = load_path.replace('.pth', '_ema.pth')
                if self.is_train and self.do_emas:
                    ema_model = self.emas[name]
                    if os.path.exists(load_path_ema):
                        load_network(load_path_ema, ema_model, self.cfg['path']['strict_load'], cfg_get(
                            self.cfg, ['path', f'pretrain_base_path_{name}']))
                    else:
                        print(
                            "WARNING! Unable to find EMA network! Starting a new EMA from given model parameters.")
                        self.emas[name] = copy.deepcopy(net)
                    if self.ema_on_cpu:
                        self.emas[name] = self.emas[name].cpu()
                if hasattr(net, 'network_loaded'):
                    net.network_loaded()
                elif hasattr(net, 'module') and hasattr(net.module, 'network_loaded'):
                    net.module.network_loaded()

    def save(self, iter_step):
        for name, net in self.networks.items():
            # Don't save non-trainable networks.
            if self.cfg['networks'][name]['trainable']:
                self.save_network(net, name, iter_step)
                if self.do_emas:
                    self.save_network(
                        self.emas[name], f'{name}_ema', iter_step)

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.cfg['path']['models'], save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        return save_path


class Trainer:
    def __init__(self,
                 cfg_path,
                 cfg):
        cfg['dist'] = False
        self.rank = -1
        if len(cfg['gpu_ids']) == 1:
            torch.cuda.set_device(cfg['gpu_ids'][0])
        print('Distributed training not possible.')
        self._profile = False
        self.val_compute_psnr = cfg_get(cfg, ['eval', 'compute_psnr'], False)
        self.val_compute_fea = cfg_get(cfg, ['eval', 'compute_fea'], False)
        self.current_step = 0
        self.iteration_rate = 0
        self.total_training_data_encountered = 0

        self.use_tqdm = True

        mkdir_and_rename(cfg['path']['experiments_root'])
        mkdirs((path for key, path in cfg['path'].items() if not key == 'experiments_root' and path is not None
                and 'pretrain_model' not in key and 'resume' not in key))

        shutil.copy(cfg_path, os.path.join(
            cfg['path']['experiments_root'],
            f'{datetime.now().strftime("%d%m%Y_%H%M%S")}_{os.path.basename(cfg_path)}'))

        # config loggers. Before it, the log will not work
        setup_logger('base', cfg['path']['log'], 'train_' +
                     cfg['name'], level=logging.INFO, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        self.logger.info(dict2str(cfg))

        # convert to NoneDict, which returns None for missing keys
        cfg = dict_to_nonedict(cfg)
        self.cfg = cfg

        # random seed
        seed = cfg_get(cfg, ['train', 'manual_seed'], default=42)
        if self.rank <= 0:
            self.logger.info(f'Random seed: {seed}')

        set_random_seed(seed)

        torch.backends.cudnn.benchmark = cfg_get(
            cfg, ['cuda_benchmarking_enabled'], True)
        torch.backends.cuda.matmul.allow_tf32 = True
        if cfg_get(cfg, ['anomaly_detection'], False):
            torch.autograd.set_detect_anomaly(True)

        # Save the compiled opt dict to the global loaded_options variable.
        # util.loaded_options = cfg
        self.train_sampler = None

        # create train and val dataloader
        dataset_ratio = 1  # enlarge the size of each epoch
        for phase, dataset_cfg in cfg['datasets'].items():
            if phase == 'train':
                self.train_set = get_dataset(dataset_cfg)
                self.dataset_debugger = get_dataset_debugger()
                train_size = int(
                    math.ceil(len(self.train_set) / dataset_cfg['batch_size']))
                total_iters = int(cfg['train']['niter'])
                self.total_epochs = int(math.ceil(total_iters / train_size))
                self.train_loader = create_dataloader(
                    self.train_set, dataset_cfg, self.train_sampler, collate_fn=None, shuffle=True)
                self.logger.info('Number of training data elements: {:,d}, iters: {:,d}'.format(
                    len(self.train_set), train_size))
                self.logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    self.total_epochs, total_iters))
            elif phase == 'val':
                if not cfg_get(cfg, ['eval', 'pure'], False):
                    continue

                self.val_set = get_dataset(dataset_cfg)
                self.val_loader = create_dataloader(
                    self.val_set, dataset_cfg, None, collate_fn=None, shuffle=False)
                self.logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_cfg['name'], len(self.val_set)))
            else:
                raise NotImplementedError(f'Phase [{phase}] is not recognized.')
        assert self.train_loader is not None

        # create model
        self.model = ExtensibleTrainer(cfg)

        # Evaluators
        self.evaluators = []
        # if 'eval' in cfg.keys() and 'evaluators' in cfg['eval'].keys():
        #     # In "pure" mode, we propagate through the normal training steps, but use validation data instead and average
        #     # the total loss. A validation dataloader is required.
        #     if cfg_get(cfg, ['eval', 'pure'], False):
        #         assert hasattr(self, 'val_loader')
        #
        #     for ev_key, ev_opt in cfg['eval']['evaluators'].items():
        #         self.evaluators.append(create_evaluator(self.model.networks[ev_opt['for']],
        #                                                 ev_opt, self.model.env))

        self.current_step = - \
            1 if 'start_step' not in cfg.keys() else cfg['start_step']
        self.total_training_data_encountered = 0 if 'training_data_encountered' not in cfg.keys(
        ) else cfg['training_data_encountered']
        self.start_epoch = 0
        if 'force_start_step' in cfg.keys():
            self.current_step = cfg['force_start_step']
            self.total_training_data_encountered = self.current_step * \
                                                   cfg['datasets']['train']['batch_size']
        cfg['current_step'] = self.current_step

        self.epoch = self.start_epoch

        # validation
        if 'val_freq' in cfg['train'].keys():
            self.val_freq = cfg['train']['val_freq'] * \
                            cfg['datasets']['train']['batch_size']
        else:
            self.val_freq = int(cfg['train']['val_freq_megasamples'] * 1000000)

        self.next_eval_step = self.total_training_data_encountered + self.val_freq
        # For whatever reason, this relieves a memory burden on the first GPU for some training sessions.

    def save(self):
        self.model.save(self.current_step)
        state = {
            'epoch': self.epoch,
            'iter': self.current_step,
            'total_data_processed': self.total_training_data_encountered
        }
        if self.dataset_debugger is not None:
            state['dataset_debugger_state'] = self.dataset_debugger.get_state()
        self.logger.info('Saving models.')

    def do_step(self, train_data):
        cfg = self.cfg
        # It may seem weird to derive this from cfg, rather than train_data. The reason this is done is
        batch_size = self.cfg['datasets']['train']['batch_size']
        # because train_data is process-local while the cfg variant represents all of the data fed across all GPUs.
        self.current_step += 1
        self.total_training_data_encountered += batch_size
        # self.current_step % cfg['logger']['print_freq'] == 0
        will_log = False

        # update learning rate
        self.model.update_learning_rate(
            self.current_step, warmup_iter=cfg['train']['warmup_iter'])

        # training
        _t = time()
        self.model.feed_data(train_data, self.current_step)
        gradient_norms_dict = self.model.optimize_parameters(
            self.current_step, return_grad_norms=will_log)
        self.iteration_rate = (time() - _t)  # / batch_size

        metrics = {}
        for s in self.model.steps:
            metrics.update(s.get_metrics())

        # log
        if self.dataset_debugger is not None:
            self.dataset_debugger.update(train_data)
        if will_log:
            # Must be run by all instances to gather consensus.
            current_model_logs = self.model.get_current_log(self.current_step)
            logs = {
                'step': self.current_step,
                'samples': self.total_training_data_encountered,
                'megasamples': self.total_training_data_encountered / 1000000,
                'iteration_rate': self.iteration_rate,
                'lr': self.model.get_current_learning_rate(),
            }
            logs.update(current_model_logs)

            if self.dataset_debugger is not None:
                logs.update(self.dataset_debugger.get_debugging_map())

            logs.update(gradient_norms_dict)
            self.logger.info(f'Training Metrics: {try_json(logs)}')

            message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(self.epoch, self.current_step)
            for v in self.model.get_current_learning_rate():
                message += '{:.3e},'.format(v)
            message += ')] '
            if cfg['wandb']:
                import wandb
                wandb_logs = {}
                for k, v in logs.items():
                    if 'histogram' in k:
                        wandb_logs[k] = wandb.Histogram(v)
                    else:
                        wandb_logs[k] = v
                if cfg_get(cfg, ['wandb_progress_use_raw_steps'], False):
                    wandb.log(wandb_logs, step=self.current_step)
                else:
                    wandb.log(wandb_logs, step=self.total_training_data_encountered)
            self.logger.info(message)

        # save models and training states
        if self.current_step > 0 and self.current_step % cfg['logger']['save_checkpoint_freq'] == 0:
            self.save()

        do_eval = self.total_training_data_encountered > self.next_eval_step
        if do_eval:
            self.next_eval_step = self.total_training_data_encountered + self.val_freq

            if cfg_get(cfg, ['eval', 'pure'], False):
                self.do_validation()
            if len(self.evaluators) != 0:
                eval_dict = {}
                for evaluator in self.evaluators:
                    eval_dict.update(evaluator.perform_eval())
                    print("Evaluator results: ", eval_dict)

        # Should not be necessary, but make absolutely sure that there is no grad leakage from validation runs.
        for net in self.model.networks.values():
            net.zero_grad()

        return metrics

    def do_validation(self):
        self.logger.info('Beginning validation.')

        metrics = []
        tq_ldr = tqdm(self.val_loader,
                      desc="Validating") if self.use_tqdm else self.val_loader

        for val_data in tq_ldr:
            self.model.feed_data(val_data, self.current_step,
                                 perform_micro_batching=False)
            metric = self.model.test()
            metrics.append(metric)
            if self.use_tqdm:
                logs = process_metrics(metrics)
                tq_ldr.set_postfix(logs, refresh=True)

        logs = process_metrics(metrics)
        logs['it'] = self.current_step
        self.logger.info(f'Validation Metrics: {json.dumps(logs)}')

    def do_training(self):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
            self.start_epoch, self.current_step))

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if self.cfg['dist']:
                self.train_sampler.set_epoch(epoch)

            metrics = []
            tq_ldr = tqdm(
                self.train_loader, desc="Training") if self.use_tqdm else self.train_loader

            _t = time()
            step = 0
            for train_data in tq_ldr:
                step = step + 1
                metric = self.do_step(train_data)
                metrics.append(metric)
                logs = process_metrics(metrics)
                logs['lr'] = self.model.get_current_learning_rate()[0]
                if self.use_tqdm:
                    tq_ldr.set_postfix(logs, refresh=True)
                logs['it'] = self.current_step
                logs['step'] = step
                logs['steps'] = len(self.train_loader)
                logs['epoch'] = self.epoch
                logs['iteration_rate'] = self.iteration_rate
                self.logger.info(f'Training Metrics: {json.dumps(logs)}')

        self.save()
        self.logger.info('Finished training!')

    def create_training_generator(self, index):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
            self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if self.cfg['dist']:
                self.train_sampler.set_epoch(epoch)

            tq_ldr = tqdm(self.train_loader, position=index)
            tq_ldr.set_description('Training')

            _t = time()
            for train_data in tq_ldr:
                yield self.model
                metric = self.do_step(train_data)
        self.save()
        self.logger.info('Finished training')
