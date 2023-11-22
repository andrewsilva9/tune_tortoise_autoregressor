"""
Created by Andrew Silva on 11/14/23
Copied from https://github.com/neonbjb/DL-Art-School with thanks to https://git.ecker.tech/mrq/ai-voice-cloning

"""

from munch import munchify
import random
import sys

import torch
import torch.nn.functional as F
import torch.utils.data

from data_utils import (load_audio, VoiceBpeTokenizer,
                        load_similar_clips, load_filepaths_and_text_type,
                        text_to_sequence, sequence_to_text)
from utils import cfg_get, create_hparams


class CharacterTokenizer:
    def encode(self, txt):
        return text_to_sequence(txt)

    def decode(self, seq):
        return sequence_to_text(seq)


class TextWavLoader(torch.utils.data.Dataset):
    def __init__(self, hparams):
        self.path = hparams['path']
        if not isinstance(self.path, list):
            self.path = [self.path]
        self.types = cfg_get(hparams, ['types'], [0 for _ in self.path])

        fetcher_mode = cfg_get(hparams, ['fetcher_mode'], 'lj')
        if not isinstance(fetcher_mode, list):
            fetcher_mode = [fetcher_mode]
        assert len(self.path) == len(fetcher_mode)

        self.load_conditioning = cfg_get(hparams, ['load_conditioning'], False)
        self.conditioning_candidates = cfg_get(
            hparams, ['num_conditioning_candidates'], 1)
        self.conditioning_length = cfg_get(
            hparams, ['conditioning_length'], 44100)
        self.debug_failures = cfg_get(
            hparams, ['debug_loading_failures'], False)
        self.load_aligned_codes = cfg_get(
            hparams, ['load_aligned_codes'], False)
        self.aligned_codes_to_audio_ratio = cfg_get(
            hparams, ['aligned_codes_ratio'], 443)
        self.audiopaths_and_text = []
        for p, fm, d_type in zip(self.path, fetcher_mode, self.types):
            self.audiopaths_and_text.extend(load_filepaths_and_text_type(p, d_type))
        self.text_cleaners = hparams.text_cleaners
        self.sample_rate = hparams.sample_rate
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
        self.max_wav_len = cfg_get(hparams, ['max_wav_length'], None)
        if self.max_wav_len is not None:
            self.max_aligned_codes = self.max_wav_len // self.aligned_codes_to_audio_ratio
        self.max_text_len = cfg_get(hparams, ['max_text_length'], None)
        assert self.max_wav_len is not None and self.max_text_len is not None
        self.use_bpe_tokenizer = cfg_get(hparams, ['use_bpe_tokenizer'], True)
        if self.use_bpe_tokenizer:
            self.tokenizer = VoiceBpeTokenizer(cfg_get(
                hparams, ['tokenizer_vocab'], '../experiments/bpe_lowercase_asr_256.json'))
        else:
            self.tokenizer = CharacterTokenizer()
        # records how many items are skipped when accessing an index.
        self.skipped_items = 0

    def get_wav_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, d_type = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        text_seq = self.get_text(text)
        wav = load_audio(audiopath, self.sample_rate)
        return (text_seq, wav, text, audiopath_and_text[0], d_type)

    def get_text(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        if self.use_bpe_tokenizer:
            # Assert if any UNK,start tokens encountered.
            assert not torch.any(tokens == 1)
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0)
        return tokens

    def __getitem__(self, index):
        self.skipped_items += 1
        try:
            tseq, wav, text, path, d_type = self.get_wav_text_pair(
                self.audiopaths_and_text[index])
            if text is None or len(text.strip()) == 0:
                raise ValueError
            if wav is None or wav.shape[-1] < (.6 * self.sample_rate):
                # Ultra short clips are also useless (and can cause problems within some models).
                raise ValueError
            cond, cond_is_self = load_similar_clips(self.audiopaths_and_text[index][0], self.conditioning_length, self.sample_rate,
                                                    n=self.conditioning_candidates) if self.load_conditioning else (None, False)
        except:
            if self.skipped_items > 100:
                raise  # Rethrow if we have nested too far.
            if self.debug_failures:
                print(
                    f"error loading {self.audiopaths_and_text[index][0]} {sys.exc_info()}")
            return self[(index+1) % len(self)]

        if self.load_aligned_codes:
            aligned_codes = self.audiopaths_and_text[index][3]

        actually_skipped_items = self.skipped_items
        self.skipped_items = 0
        if wav is None or \
            (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len) or \
                (self.max_text_len is not None and tseq.shape[0] > self.max_text_len):
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            if self.debug_failures:
                print(
                    f"error loading {path}: ranges are out of bounds; {wav.shape[-1]}, {tseq.shape[0]}")
            rv = random.randint(0, len(self)-1)
            return self[rv]
        orig_output = wav.shape[-1]
        orig_text_len = tseq.shape[0]
        if wav.shape[-1] != self.max_wav_len:
            wav = F.pad(wav, (0, self.max_wav_len - wav.shape[-1]))
            if self.load_aligned_codes:
                # These codes are aligned to audio inputs, so make sure to pad them as well.
                aligned_codes = F.pad(
                    aligned_codes, (0, self.max_aligned_codes-aligned_codes.shape[0]))
        if tseq.shape[0] != self.max_text_len:
            tseq = F.pad(tseq, (0, self.max_text_len - tseq.shape[0]))
        res = {
            'real_text': text,
            'padded_text': tseq,
            'text_lengths': torch.tensor(orig_text_len, dtype=torch.long),
            'wav': wav,
            'wav_lengths': torch.tensor(orig_output, dtype=torch.long),
            'filenames': path,
            'skipped_items': actually_skipped_items,
            'type': d_type,
        }
        if self.load_conditioning:
            res['conditioning'] = cond
            res['conditioning_contains_self'] = cond_is_self
        if self.load_aligned_codes:
            res['aligned_codes'] = aligned_codes
        return res

    def __len__(self):
        return len(self.audiopaths_and_text)


class PairedVoiceDebugger:
    def __init__(self):
        self.total_items = 0
        self.loaded_items = 0
        self.self_conditioning_items = 0

    def get_state(self):
        return {'total_items': self.total_items,
                'loaded_items': self.loaded_items,
                'self_conditioning_items': self.self_conditioning_items}

    def load_state(self, state):
        if isinstance(state, dict):
            self.total_items = opt_get(state, ['total_items'], 0)
            self.loaded_items = opt_get(state, ['loaded_items'], 0)
            self.self_conditioning_items = opt_get(
                state, ['self_conditioning_items'], 0)

    def update(self, batch):
        self.total_items += batch['wav'].shape[0]
        self.loaded_items += batch['skipped_items'].sum().item()
        if 'conditioning' in batch.keys():
            self.self_conditioning_items += batch['conditioning_contains_self'].sum(
            ).item()

    def get_debugging_map(self):
        return {
            'total_samples_loaded': self.total_items,
            'percent_skipped_samples': (self.loaded_items - self.total_items) / self.loaded_items,
            'percent_conditioning_is_self': self.self_conditioning_items / self.loaded_items,
        }


def get_dataset(dataset_cfg):
    default_params = create_hparams()
    default_params.update(dataset_cfg)
    dataset_opt = munchify(default_params)
    dataset = TextWavLoader(dataset_opt)
    return dataset


def get_dataset_debugger():
    return PairedVoiceDebugger()


def create_dataloader(dataset, dataset_cfg, sampler=None, collate_fn=None, shuffle=True):
    phase = dataset_cfg['phase']
    pin_memory = dataset_cfg.get('pin_memory', True)
    if phase == 'train':
        num_workers = dataset_cfg['n_workers']
        batch_size = dataset_cfg['batch_size']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=pin_memory, collate_fn=collate_fn, persistent_workers=num_workers > 0)
    else:
        batch_size = dataset_cfg['batch_size'] or 1
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                           pin_memory=pin_memory, collate_fn=collate_fn)
