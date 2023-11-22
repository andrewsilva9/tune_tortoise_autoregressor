"""
Created by Andrew Silva on 11/14/23
Copied from https://github.com/neonbjb/DL-Art-School with thanks to https://git.ecker.tech/mrq/ai-voice-cloning

"""
import os
import re
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from audio2numpy import open_audio
from scipy.io.wavfile import read
from utils import SYMBOLS
from unidecode import unidecode
import inflect

from tokenizers import Tokenizer


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png',
                  '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.webp', '.WEBP']
_symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
_id_to_symbol = {i: s for i, s in enumerate(SYMBOLS)}
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path, qualifier=is_image_file):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if qualifier(fname) and 'ref.jpg' not in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    if not images:
        print("Warning: {:s} has no valid image file".format(path))
    return images


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.
    else:
        raise NotImplemented(
            f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def find_files_of_type(data_type, dataroot, weights=None, qualifier=is_image_file):
    if weights is None:
        weights = []
    if isinstance(dataroot, list):
        paths = []
        for i in range(len(dataroot)):
            r = dataroot[i]
            extends = 1

            # Weights have the effect of repeatedly adding the paths from the given root to the final product.
            if weights:
                extends = weights[i]
            for j in range(extends):
                paths.extend(_get_paths_from_images(r, qualifier))
        paths = sorted(paths)
        sizes = len(paths)
    else:
        paths = sorted(_get_paths_from_images(dataroot, qualifier))
        sizes = len(paths)
    return paths, sizes

def is_audio_file(filename):
    AUDIO_EXTENSIONS = ['.wav', '.mp3', '.wma', '.m4b', '.flac', '.aac']
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def load_audio(audiopath, sampling_rate):
    if audiopath[-4:] == '.wav':
        audio, lsr = load_wav_to_torch(audiopath)
    elif audiopath[-4:] == '.mp3':
        # https://github.com/neonbjb/pyfastmp3decoder  - Definitely worth it.
        from pyfastmp3decoder.mp3decoder import load_mp3
        audio, lsr = load_mp3(audiopath, sampling_rate)
        audio = torch.FloatTensor(audio)
    else:
        audio, lsr = open_audio(audiopath)
        audio = torch.FloatTensor(audio)

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)


def load_similar_clips(path, sample_length, sample_rate, n=3, fallback_to_self=True):
    sim_path = os.path.join(os.path.dirname(path), 'similarities.pth')
    candidates = []
    if os.path.exists(sim_path):
        similarities = torch.load(sim_path)
        fname = os.path.basename(path)
        if fname in similarities.keys():
            candidates = [os.path.join(os.path.dirname(path), s)
                          for s in similarities[fname]]
        else:
            print(
                f'Similarities list found for {path} but {fname} was not in that list.')
        # candidates.append(path)  # Always include self as a possible similar clip.
    if len(candidates) == 0:
        if fallback_to_self:
            candidates = [path]
        else:
            candidates = find_files_of_type(
                'img', os.path.dirname(path), qualifier=is_audio_file)[0]

    # Sanity check to ensure we aren't loading "related files" that aren't actually related.
    assert len(candidates) < 50000
    if len(candidates) == 0:
        print(f"No conditioning candidates found for {path}")
        raise NotImplementedError()

    # Sample with replacement. This can get repeats, but more conveniently handles situations where there are not enough candidates.
    related_clips = []
    contains_self = False
    for k in range(n):
        rel_path = random.choice(candidates)
        contains_self = contains_self or (rel_path == path)
        rel_clip = load_audio(rel_path, sample_rate)
        gap = rel_clip.shape[-1] - sample_length
        if gap < 0:
            rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
        elif gap > 0:
            rand_start = random.randint(0, gap)
            rel_clip = rel_clip[:, rand_start:rand_start+sample_length]
        related_clips.append(rel_clip)
    if n > 1:
        return torch.stack(related_clips, dim=0), contains_self
    else:
        return related_clips[0], contains_self


def load_filepaths_and_text_type(filename, type, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [
            list(line.strip().split(split)) + [type] for line in f]
        base = os.path.dirname(filename)
        for j in range(len(filepaths_and_text)):
            filepaths_and_text[j][0] = os.path.join(
                base, filepaths_and_text[j][0])
    return filepaths_and_text


def text_to_sequence(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence

      Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(english_cleaners(text))
            break
        sequence += _symbols_to_sequence(english_cleaners(text))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ''
    for symbol_id in sequence:
        if isinstance(symbol_id, torch.Tensor):
            symbol_id = symbol_id.item()
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'


# Character cleaning and turning into pronunciations

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

WHITESPACE_RE = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
ABBREVIATIONS = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in ABBREVIATIONS:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(WHITESPACE_RE, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    text = text.replace('"', '')
    return text


class VoiceBpeTokenizer:
    def __init__(self, vocab_file, preprocess=None):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        self.language = vocab['model']['language'] if 'language' in vocab['model'] else None

        if preprocess is None:
            self.preprocess = 'pre_tokenizer' in vocab and vocab['pre_tokenizer']
        else:
            self.preprocess = preprocess

        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt):
        return english_cleaners(txt)

    def encode(self, txt):
        if self.preprocess:
            txt = self.preprocess_text(txt)
        txt = txt.replace(' ', '[SPACE]')
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(
            seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')
        return txt
