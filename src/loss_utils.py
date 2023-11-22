"""
Created by Andrew Silva on 11/17/23
Copied from https://github.com/neonbjb/DL-Art-School with thanks to https://git.ecker.tech/mrq/ai-voice-cloning

"""
import torch
import torch.nn as nn
from utils import cfg_get


def create_loss(loss_cfg, env):
    return DirectLoss(loss_cfg, env)


class DirectLoss(nn.Module):
    def __init__(self, cfg, env):
        super(DirectLoss, self).__init__()
        self.env = env
        self.cfg = cfg
        self.inverted = cfg['inverted'] if 'inverted' in cfg.keys() else False
        self.key = cfg['key']
        self.anneal = cfg_get(cfg, ['annealing_termination_step'], 0)

    def forward(self, _, state):
        if self.inverted:
            loss = -torch.mean(state[self.key])
        else:
            loss = torch.mean(state[self.key])
        if self.anneal > 0:
            loss = loss * (1 - (self.anneal - min(self.env['step'], self.anneal)) / self.anneal)
        return loss


class LossAccumulator:
    def __init__(self, buffer_sz=50):
        self.buffer_sz = buffer_sz
        self.buffers = {}
        self.counters = {}

    def add_loss(self, name, tensor):
        if name not in self.buffers.keys():
            if "_histogram" in name:
                tensor = torch.flatten(tensor.detach().cpu())
                self.buffers[name] = (0, torch.zeros(
                    (self.buffer_sz, tensor.shape[0])), False)
            else:
                self.buffers[name] = (0, torch.zeros(self.buffer_sz), False)
        i, buf, filled = self.buffers[name]
        # Can take tensors or just plain python numbers.
        if '_histogram' in name:
            buf[i] = torch.flatten(tensor.detach().cpu())
        elif isinstance(tensor, torch.Tensor):
            buf[i] = tensor.detach().cpu()
        else:
            buf[i] = tensor
        filled = i+1 >= self.buffer_sz or filled
        self.buffers[name] = ((i+1) % self.buffer_sz, buf, filled)

    def increment_metric(self, name):
        if name not in self.counters.keys():
            self.counters[name] = 1
        else:
            self.counters[name] += 1

    def as_dict(self):
        result = {}
        for k, v in self.buffers.items():
            i, buf, filled = v
            if '_histogram' in k:
                result["loss_" + k] = torch.flatten(buf)
            if filled:
                result["loss_" + k] = torch.mean(buf)
            else:
                result["loss_" + k] = torch.mean(buf[:i])
        for k, v in self.counters.items():
            result[k] = v
        return result

