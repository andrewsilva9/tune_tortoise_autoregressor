"""
Created by Andrew Silva on 11/16/23
Copied from https://github.com/neonbjb/DL-Art-School with thanks to https://git.ecker.tech/mrq/ai-voice-cloning

"""
import functools
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import VectorQuantize

from math import sqrt

from utils import cfg_get, Loader


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, new_return_order=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.codes = None
        self.new_return_order = new_return_order

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input_data, return_soft_codes=False):
        flatten = input_data.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        soft_codes = -dist
        _, embed_ind = soft_codes.max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input_data.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input_data).pow(2).mean()
        quantize = input_data + (quantize - input_data).detach()

        if return_soft_codes:
            return quantize, diff, embed_ind, soft_codes.view(input_data.shape[:-1] + (-1,))
        elif self.new_return_order:
            return quantize, embed_ind, diff
        else:
            return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, chan, conv, activation):
        super().__init__()
        self.net = nn.Sequential(
            conv(chan, chan, 3, padding=1),
            activation(),
            conv(chan, chan, 3, padding=1),
            activation(),
            conv(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class UpsampledConv(nn.Module):
    def __init__(self, conv, *args, **kwargs):
        super().__init__()
        assert 'stride' in kwargs.keys()
        self.stride = kwargs['stride']
        del kwargs['stride']
        self.conv = conv(*args, **kwargs)

    def forward(self, x):
        up = nn.functional.interpolate(
            x, scale_factor=self.stride, mode='nearest')
        return self.conv(up)


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        positional_dims=2,
        num_tokens=512,
        codebook_dim=512,
        num_layers=3,
        num_resnet_blocks=0,
        hidden_dim=64,
        channels=3,
        stride=2,
        kernel_size=4,
        use_transposed_convs=True,
        encoder_norm=False,
        activation='relu',
        smooth_l1_loss=False,
        straight_through=False,
        normalization=None,  # ((0.5,) * 3, (0.5,) * 3),
        record_codes=False,
        use_lr_quantizer=False,
        lr_quantizer_args=None,
    ):
        super().__init__()
        if lr_quantizer_args is None:
            lr_quantizer_args = {}
        has_resblocks = num_resnet_blocks > 0

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.straight_through = straight_through
        self.positional_dims = positional_dims

        # This VAE only supports 1d and 2d inputs for now.
        assert positional_dims > 0 and positional_dims < 3
        if positional_dims == 2:
            conv = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        else:
            conv = nn.Conv1d
            conv_transpose = nn.ConvTranspose1d
        if not use_transposed_convs:
            conv_transpose = functools.partial(UpsampledConv, conv)

        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'silu':
            act = nn.SiLU
        else:
            assert NotImplementedError()

        enc_layers = []
        dec_layers = []

        if num_layers > 0:
            enc_chans = [hidden_dim * 2 ** i for i in range(num_layers)]
            dec_chans = list(reversed(enc_chans))

            enc_chans = [channels, *enc_chans]

            dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
            dec_chans = [dec_init_chan, *dec_chans]

            enc_chans_io, dec_chans_io = map(lambda t: list(
                zip(t[:-1], t[1:])), (enc_chans, dec_chans))

            pad = (kernel_size - 1) // 2
            for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
                enc_layers.append(nn.Sequential(
                    conv(enc_in, enc_out, kernel_size, stride=stride, padding=pad), act()))
                if encoder_norm:
                    enc_layers.append(nn.GroupNorm(8, enc_out))
                dec_layers.append(nn.Sequential(conv_transpose(
                    dec_in, dec_out, kernel_size, stride=stride, padding=pad), act()))
            dec_out_chans = dec_chans[-1]
            innermost_dim = dec_chans[0]
        else:
            enc_layers.append(nn.Sequential(
                conv(channels, hidden_dim, 1), act()))
            dec_out_chans = hidden_dim
            innermost_dim = hidden_dim

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(innermost_dim, conv, act))
            enc_layers.append(ResBlock(innermost_dim, conv, act))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, conv(codebook_dim, innermost_dim, 1))

        enc_layers.append(conv(innermost_dim, codebook_dim, 1))
        dec_layers.append(conv(dec_out_chans, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

        if use_lr_quantizer:
            self.codebook = VectorQuantize(
                dim=codebook_dim, codebook_size=num_tokens, **lr_quantizer_args)
        else:
            self.codebook = Quantize(
                codebook_dim, num_tokens, new_return_order=True)

        # take care of normalization within class
        self.normalization = normalization
        self.record_codes = record_codes
        if record_codes:
            self.codes = torch.zeros((1228800,), dtype=torch.long)
            self.code_ind = 0
            self.total_codes = 0
        self.internal_step = 0

    def norm(self, images):
        if not self.normalization is not None:
            return images

        means, stds = map(lambda t: torch.as_tensor(
            t).to(images), self.normalization)
        arrange = 'c -> () c () ()' if self.positional_dims == 2 else 'c -> () c ()'
        means, stds = map(lambda t: rearrange(t, arrange), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    def get_debug_values(self, step, __):
        if self.record_codes and self.total_codes > 0:
            # Report annealing schedule
            return {'histogram_codes': self.codes[:self.total_codes]}
        else:
            return {}

    @torch.no_grad()
    def get_codebook_indices(self, images):
        img = self.norm(images)
        logits = self.encoder(img).permute(
            (0, 2, 3, 1) if len(img.shape) == 4 else (0, 2, 1))
        sampled, codes, _ = self.codebook(logits)
        self.log_codes(codes)
        return codes

    def decode(
        self,
        img_seq
    ):
        self.log_codes(img_seq)
        if hasattr(self.codebook, 'embed_code'):
            image_embeds = self.codebook.embed_code(img_seq)
        else:
            image_embeds = F.embedding(img_seq, self.codebook.codebook)
        b, n, d = image_embeds.shape

        kwargs = {}
        if self.positional_dims == 1:
            arrange = 'b n d -> b d n'
        else:
            h = w = int(sqrt(n))
            arrange = 'b (h w) d -> b d h w'
            kwargs = {'h': h, 'w': w}
        image_embeds = rearrange(image_embeds, arrange, **kwargs)
        images = [image_embeds]
        for layer in self.decoder:
            images.append(layer(images[-1]))
        return images[-1], images[-2]

    def infer(self, img):
        img = self.norm(img)
        logits = self.encoder(img).permute(
            (0, 2, 3, 1) if len(img.shape) == 4 else (0, 2, 1))
        sampled, codes, commitment_loss = self.codebook(logits)
        return self.decode(codes)

    # Note: This module is not meant to be run in forward() except while training. It has special logic which performs
    # evaluation using quantized values when it detects that it is being run in eval() mode, which will be substantially
    # more lossy (but useful for determining network performance).
    def forward(
        self,
        img
    ):
        img = self.norm(img)
        logits = self.encoder(img).permute(
            (0, 2, 3, 1) if len(img.shape) == 4 else (0, 2, 1))
        sampled, codes, commitment_loss = self.codebook(logits)
        sampled = sampled.permute(
            (0, 3, 1, 2) if len(img.shape) == 4 else (0, 2, 1))

        if self.training:
            out = sampled
            for d in self.decoder:
                out = d(out)
            self.log_codes(codes)
        else:
            # This is non-differentiable, but gives a better idea of how the network is actually performing.
            out, _ = self.decode(codes)

        # reconstruction loss
        recon_loss = self.loss_fn(img, out, reduction='none')

        return recon_loss, commitment_loss, out

    def log_codes(self, codes):
        # This is so we can debug the distribution of codes being learned.
        if self.record_codes and self.internal_step % 10 == 0:
            codes = codes.flatten()
            l = codes.shape[0]
            i = self.code_ind if (
                self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i:i+l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
            self.total_codes += 1
        self.internal_step += 1


def get_discrete_vae(opt_net):
    return DiscreteVAE(**cfg_get(opt_net, ['kwargs'], {}))


def load_dvae_from_config(cfg_file=None, model_name=None, also_load_savepoint=True, load_path=None,
                          preloaded_options=None, strict_load=True, device=None):
    if preloaded_options is not None:
        opt = preloaded_options
    else:
        with open(cfg_file, mode='r') as f:
            opt = yaml.load(f, Loader=Loader)
    if model_name is None:
        model_cfg = opt['networks'].values()
        model_name = next(opt['networks'].keys())
    else:
        model_cfg = opt['networks'][model_name]
    if 'which_model_G' in model_cfg.keys() and 'which_model' not in model_cfg.keys():
        model_cfg['which_model'] = model_cfg['which_model_G']
    model = get_discrete_vae(model_cfg).to(device)
    if also_load_savepoint and f'pretrain_model_{model_name}' in opt['path'].keys():
        assert load_path is None
        load_path = opt['path'][f'pretrain_model_{model_name}']
    if load_path is not None:
        print(f"Loading from {load_path}")
        sd = torch.load(load_path, map_location=device)
        model.load_state_dict(sd, strict=strict_load)
    return model
