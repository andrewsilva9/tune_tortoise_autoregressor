"""
Created by Andrew Silva on 11/13/23
Copied from https://git.ecker.tech/mrq/ai-voice-cloning and https://github.com/neonbjb/DL-Art-School

"""

from collections import OrderedDict
import os
from datetime import datetime
import random
import numpy as np
import logging
import torch
try:
    # 1.13.1
    from torch._six import inf
except Exception as e:
    # 2.0
    from torch import inf

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

import os.path as osp

import yaml

SYMBOLS = ['_', '-', '!', "'", '(', ')', ',', '.', ':', ';', '?', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '@AA', '@AA0', '@AA1', '@AA2', '@AE', '@AE0', '@AE1', '@AE2', '@AH', '@AH0', '@AH1', '@AH2', '@AO', '@AO0', '@AO1', '@AO2', '@AW', '@AW0', '@AW1', '@AW2', '@AY', '@AY0', '@AY1', '@AY2', '@B', '@CH', '@D', '@DH', '@EH', '@EH0', '@EH1', '@EH2', '@ER', '@ER0', '@ER1', '@ER2', '@EY', '@EY0', '@EY1', '@EY2', '@F', '@G', '@HH', '@IH', '@IH0', '@IH1', '@IH2', '@IY', '@IY0', '@IY1', '@IY2', '@JH', '@K', '@L', '@M', '@N', '@NG', '@OW', '@OW0', '@OW1', '@OW2', '@OY', '@OY0', '@OY1', '@OY2', '@P', '@R', '@S', '@SH', '@T', '@TH', '@UH', '@UH0', '@UH1', '@UH2', '@UW', '@UW0', '@UW1', '@UW2', '@V', '@W', '@Y', '@Z', '@ZH']



def OrderedYaml():
    """yaml orderedDict support"""
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(cfg_path, is_train=True):
    yaml_loader, yaml_dumper = OrderedYaml()

    with open(cfg_path, mode='r') as f:
        cfg = yaml.load(f, Loader=yaml_loader)

    cfg['is_train'] = is_train

    # datasets
    if 'datasets' in cfg.keys():
        for phase, dataset in cfg['datasets'].items():
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            is_lmdb = False
            dataset['data_type'] = 'img'
            if dataset['mode'].endswith('mc'):  # for memcached
                dataset['data_type'] = 'mc'
                dataset['mode'] = dataset['mode'].replace('_mc', '')

    # path
    if 'path' in cfg.keys():
        for key, path in cfg['path'].items():
            if path and key in cfg['path'] and (key not in ['strict_load', 'optimizer_reset']):
                cfg['path'][key] = osp.expanduser(path)
    else:
        cfg['path'] = {}
    cfg['path']['root'] = cfg_path.split('/')[0] + '/'
    if is_train:
        experiments_root = osp.join(
            cfg['path']['root'], 'training', cfg['name'], "finetune")
        cfg['path']['experiments_root'] = experiments_root
        cfg['path']['models'] = osp.join(experiments_root, 'models')
        cfg['path']['training_state'] = osp.join(
            experiments_root, 'training_state')
        cfg['path']['log'] = experiments_root
        cfg['path']['val_images'] = osp.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in cfg['name']:
            cfg['train']['val_freq'] = 8
            cfg['logger']['print_freq'] = 1
            cfg['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(cfg['path']['root'], 'results', cfg['name'])
        cfg['path']['results_root'] = results_root
        cfg['path']['log'] = results_root

    return cfg


def cfg_get(cfg, keys, default=None):
    """
    Get an item from a config, using a list of keys
    e.g., cfg = {'model': {'learning_rate': 1e-5}} -- cfg_get(cfg, ['model', 'learning_rate'], default=1e-4) -> 1e-5
    """
    #
    if isinstance(keys, str):
        keys = [keys]
    if cfg is None:
        return default
    ret = cfg
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = dict(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        # When different from sampling_rate, dataset automatically interpolates to sampling_rate
        input_sample_rate=22050,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,  # This means a MEL is 1/256th the equivalent audio.
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(SYMBOLS),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
        mask_padding=True  # set model's padded outputs to padded values
    )

    return hparams


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + datetime.now().strftime('%y%m%d-%H%M%S')
        print(f'Path already exists. Rename it to [{new_name}]')
        logger = logging.getLogger('base')
        logger.info(f'Path already exists. Rename it to [{new_name}]')
        os.rename(path, new_name)
    os.makedirs(path)


def map_cuda_to_correct_device(storage, loc):
    if str(loc).startswith('cuda'):
        return storage.cuda(torch.cuda.current_device())
    else:
        return storage.cpu()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def dict2str(cfg, indent_l=1):
    """dict to string for logger"""
    msg = ''
    for k, v in cfg.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + f'_{datetime.now().strftime("%y%m%d-%H%M%S")}.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(cfg):
    if isinstance(cfg, dict):
        new_opt = dict()
        for key, sub_opt in cfg.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(cfg, list):
        return [dict_to_nonedict(sub_cfg) for sub_cfg in cfg]
    else:
        return cfg


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_grad_norm(parameters: list, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(
            norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(
            p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


def load_network(load_path, network, strict=True, pretrain_base_path=None):
    load_net = torch.load(
        load_path, map_location=map_cuda_to_correct_device)

    # Support loading torch.save()s for whole models as well as just state_dicts.
    if 'state_dict' in load_net:
        load_net = load_net['state_dict']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'

    if pretrain_base_path is not None:
        t = load_net
        load_net = {}
        for k, v in t.items():
            if k.startswith(pretrain_base_path):
                load_net[k[len(pretrain_base_path):]] = v
    for k, v in load_net.items():
        if k.startswith('module.'):
            k = k.replace('.module', '')
        load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
