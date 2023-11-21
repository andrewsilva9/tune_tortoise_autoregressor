"""
Created by Andrew Silva on 11/16/23
Copied from https://git.ecker.tech/mrq/ai-voice-cloning and https://github.com/neonbjb/DL-Art-School
"""
import torch
import torchaudio
from torch.cuda.amp import autocast

from utils import cfg_get
from dvae_network import get_discrete_vae, load_dvae_from_config

MEL_MIN = -11.512925148010254
TORCH_MEL_MAX = 4.82  # FYI: this STILL isn't assertive enough...


def normalize_torch_mel(mel):
    return 2 * ((mel - MEL_MIN) / (TORCH_MEL_MAX - MEL_MIN)) - 1


def extract_params_from_state(params: object, state: object, root: object = True) -> object:
    if isinstance(params, list) or isinstance(params, tuple):
        p = [extract_params_from_state(r, state, False) for r in params]
    elif isinstance(params, str):
        if params == 'None':
            p = None
        else:
            p = state[params]
    else:
        p = params
    # The root return must always be a list.
    if root and not isinstance(p, list):
        p = [p]
    return p


class Injector(torch.nn.Module):
    def __init__(self, cfg, env):
        super(Injector, self).__init__()
        self.cfg = cfg
        self.env = env
        if 'in' in cfg.keys():
            self.input = cfg['in']
        if 'out' in cfg.keys():
            self.output = cfg['out']

    # This should return a dict of new state variables.
    def forward(self, state):
        raise NotImplementedError


# Uses a generator to synthesize an image from [in] and injects the results into [out]
# Note that results are *not* detached.
class GeneratorInjector(Injector):
    def __init__(self, cfg, env):
        super(GeneratorInjector, self).__init__(cfg, env)
        self.grad = cfg['grad'] if 'grad' in cfg.keys() else True
        # If specified, this method is called instead of __call__()
        self.method = cfg_get(cfg, ['method'], None)
        self.args = cfg_get(cfg, ['args'], {})
        self.fp16_override = cfg_get(cfg, ['fp16'], True)

    def forward(self, state):
        gen = self.env['generators'][self.cfg['generator']]

        if self.method is not None and hasattr(gen, 'module'):
            gen = gen.module  # Dereference DDP wrapper.
        method = gen if self.method is None else getattr(gen, self.method)

        with autocast(enabled=self.env['cfg']['fp16'] and self.fp16_override):
            if isinstance(self.input, list):
                params = extract_params_from_state(self.input, state)
            else:
                params = [state[self.input]]
            if self.grad:
                results = method(*params, **self.args)
            else:
                was_training = gen.training
                gen.eval()
                with torch.no_grad():
                    results = method(*params, **self.args)
                if was_training:
                    gen.train()
        new_state = {}
        if isinstance(self.output, list):
            # Only dereference tuples or lists, not tensors. IF YOU REACH THIS ERROR, REMOVE THE BRACES AROUND YOUR OUTPUTS IN THE YAML CONFIG
            assert isinstance(results, list) or isinstance(results, tuple)
            for i, k in enumerate(self.output):
                new_state[k] = results[i]
        else:
            new_state[self.output] = results

        return new_state


class DiscreteTokenInjector(Injector):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        dvae_cfg = cfg_get(
            cfg, ['dvae_config'], "../experiments/train_diffusion_vocoder_22k_level.yml")
        dvae_name = cfg_get(cfg, ['dvae_name'], 'dvae')
        self.dvae = load_dvae_from_config(
            dvae_cfg, dvae_name, device=f'cuda:{env["device"]}').eval()

    def forward(self, state):
        inp = state[self.input]
        with torch.no_grad():
            self.dvae = self.dvae.to(inp.device)
            codes = self.dvae.get_codebook_indices(inp)
            return {self.output: codes}


class TorchMelSpectrogramInjector(Injector):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = cfg_get(cfg, ['filter_length'], 1024)
        self.hop_length = cfg_get(cfg, ['hop_length'], 256)
        self.win_length = cfg_get(cfg, ['win_length'], 1024)
        self.n_mel_channels = cfg_get(cfg, ['n_mel_channels'], 80)
        self.mel_fmin = cfg_get(cfg, ['mel_fmin'], 0)
        self.mel_fmax = cfg_get(cfg, ['mel_fmax'], 8000)
        self.sampling_rate = cfg_get(cfg, ['sampling_rate'], 22050)
        norm = cfg_get(cfg, ['normalize'], False)
        self.true_norm = cfg_get(cfg, ['true_normalization'], False)
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=norm,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = cfg_get(cfg, ['mel_norm_file'], None)
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, state):
        with torch.no_grad():
            inp = state[self.input]
            # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            if len(inp.shape) == 3:
                inp = inp.squeeze(1)
            assert len(inp.shape) == 2
            self.mel_stft = self.mel_stft.to(inp.device)
            mel = self.mel_stft(inp)
            # Perform dynamic range compression
            mel = torch.log(torch.clamp(mel, min=1e-5))
            if self.mel_norms is not None:
                self.mel_norms = self.mel_norms.to(mel.device)
                mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
            if self.true_norm:
                mel = normalize_torch_mel(mel)
            return {self.output: mel}


class ForEachInjector(Injector):
    def __init__(self, cfg, env):
        super(ForEachInjector, self).__init__(cfg, env)
        o = cfg.copy()
        o['type'] = cfg['subtype']
        o['in'] = '_in'
        o['out'] = '_out'
        # TODO: self.injector is a torch mel
        # self.injector = create_injector(o, self.env)  # torch mel
        self.injector = get_injector(o, self.env)
        self.aslist = cfg['aslist'] if 'aslist' in cfg.keys() else False

    def forward(self, state):
        injs = []
        st = state.copy()
        inputs = state[self.cfg['in']]
        for i in range(inputs.shape[1]):
            st['_in'] = inputs[:, i]
            injs.append(self.injector(st)['_out'])
        if self.aslist:
            return {self.output: injs}
        else:
            return {self.output: torch.stack(injs, dim=1)}


def get_injector(injector_details, env):
    i_name = injector_details['type']
    if i_name == 'for_each':
        return ForEachInjector(injector_details, env)
    elif i_name == 'torch_mel_spectrogram':
        return TorchMelSpectrogramInjector(injector_details, env)
    elif i_name == 'discrete_token':
        return DiscreteTokenInjector(injector_details, env)
    elif i_name == 'generator':
        return GeneratorInjector(injector_details, env)
