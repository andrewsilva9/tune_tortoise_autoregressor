"""
Created by Andrew Silva on 11/20/23
Copied from https://github.com/neonbjb/tortoise-tts

"""
from tortoise.api import TextToSpeech, load_discrete_vocoder_diffuser, fix_autoregressive_output, do_spectrogram_diffusion
from tortoise.utils.audio import load_audio
from utils import load_network
from unified_voice_network import UnifiedVoice
import torch
import os
import random
from time import time
import torch.nn.functional as F
from tqdm import tqdm


class CustomTTS(TextToSpeech):
    """
    Main entry point into Tortoise.
    """

    def __init__(self,
                 custom_autoregressive_path='../training/lich_king/models/finetune/255_gpt.pth', **kwargs):
        """
        Constructor
        :param autoregressive_batch_size: Specifies how many samples to generate per batch. Lower this if you are seeing
                                          GPU OOM errors. Larger numbers generates slightly faster.
        :param models_dir: Where model weights are stored. This should only be specified if you are providing your own
                           models, otherwise use the defaults.
        :param enable_redaction: When true, text enclosed in brackets are automatically redacted from the spoken output
                                 (but are still rendered by the model). This can be used for prompt engineering.
                                 Default is true.
        :param device: Device to use when running the model. If omitted, the device will be automatically chosen.
        """
        super().__init__(**kwargs)
        self.AR_model_home = self.models_dir.split('/')[-1]
        self.load_autoregressive_model(custom_autoregressive_path)

    def load_autoregressive_model(self, autoregressive_model_path):
        self.loading = True
        print(f"Loading autoregressive model: {autoregressive_model_path}")

        mdl_kwargs = {
            "layers": 30,
            "model_dim": 1024,
            "heads": 16,
            "max_text_tokens": 402,
            "max_mel_tokens": 604,
            "max_conditioning_inputs": 2,
            "mel_length_compression": 1024,
            "number_text_tokens": 256,
            "number_mel_codes": 8194,
            "start_mel_token": 8192,
            "stop_mel_token": 8193,
            "start_text_token": 255,
            "train_solo_embeddings": False,
            "use_mel_codes_as_input": True,
            "checkpointing": True,
            "tortoise_compat": True,
        }

        self.autoregressive = UnifiedVoice(**mdl_kwargs).cpu().eval()
        self.autoregressive.load_state_dict(torch.load(autoregressive_model_path))
        self.AR_model_home = autoregressive_model_path.split('/')[-1]
        self.loading = False
        print(f"Loaded autoregressive model")

    def load_custom_model(self, model_path, strict=True):
        self.loading = True
        load_network(model_path, self.autoregressive, strict=strict)
        self.loading = False
        print(f"Loaded autoregressive model from {model_path}")
        self.AR_model_home = model_path.split('/')[-1]

    def tts(self, text, voice_samples=None, conditioning_latents=None, k=1, verbose=True, use_deterministic_seed=None,
            return_deterministic_state=False,
            # autoregressive generation parameters follow
            num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8,
            max_mel_tokens=500,
            # CVVP parameters follow
            cvvp_amount=.0,
            # diffusion generation parameters follow
            diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0,
            **hf_generate_kwargs):
        """
        Produces an audio clip of the given text being spoken with the given reference voice.
        :param text: Text to be spoken.
        :param voice_samples: List of 2 or more ~10 second reference clips which should be torch tensors containing 22.05kHz waveform data.
        :param conditioning_latents: A tuple of (autoregressive_conditioning_latent, diffusion_conditioning_latent), which
                                     can be provided in lieu of voice_samples. This is ignored unless voice_samples=None.
                                     Conditioning latents can be retrieved via get_conditioning_latents().
        :param k: The number of returned clips. The most likely (as determined by Tortoises' CLVP model) clips are returned.
        :param verbose: Whether or not to print log messages indicating the progress of creating a clip. Default=true.
        ~~AUTOREGRESSIVE KNOBS~~
        :param num_autoregressive_samples: Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
               As Tortoise is a probabilistic model, more samples means a higher probability of creating something "great".
        :param temperature: The softmax temperature of the autoregressive model.
        :param length_penalty: A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs.
        :param repetition_penalty: A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence
                                   of long silences or "uhhhhhhs", etc.
        :param top_p: P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely" (aka boring) outputs.
        :param max_mel_tokens: Restricts the output length. (0,600] integer. Each unit is 1/20 of a second.
        :param typical_sampling: Turns typical sampling on or off. This sampling mode is discussed in this paper: https://arxiv.org/abs/2202.00666
                                 I was interested in the premise, but the results were not as good as I was hoping. This is off by default, but
                                 could use some tuning.
        :param typical_mass: The typical_mass parameter from the typical_sampling algorithm.
        ~~CLVP-CVVP KNOBS~~
        :param cvvp_amount: Controls the influence of the CVVP model in selecting the best output from the autoregressive model.
                            [0,1]. Values closer to 1 mean the CVVP model is more important, 0 disables the CVVP model.
        ~~DIFFUSION KNOBS~~
        :param diffusion_iterations: Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine
                                     the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better,
                                     however.
        :param cond_free: Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs two forward passes for
                          each diffusion step: one with the outputs of the autoregressive model and one with no conditioning priors. The output
                          of the two is blended according to the cond_free_k value below. Conditioning-free diffusion is the real deal, and
                          dramatically improves realism.
        :param cond_free_k: Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf].
                            As cond_free_k increases, the output becomes dominated by the conditioning-free signal.
                            Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k
        :param diffusion_temperature: Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
                                      are the "mean" prediction of the diffusion network and will sound bland and smeared.
        ~~OTHER STUFF~~
        :param hf_generate_kwargs: The huggingface Transformers generate API is used for the autoregressive transformer.
                                   Extra keyword args fed to this function get forwarded directly to that API. Documentation
                                   here: https://huggingface.co/docs/transformers/internal/generation_utils
        :return: Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
                 Sample rate is 24kHz.
        """
        deterministic_seed = self.deterministic_state(seed=use_deterministic_seed)

        text_tokens = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
        assert text_tokens.shape[
                   -1] < 400, 'Too much text provided. Break the text up into separate segments and re-try inference.'
        auto_conds = None
        if voice_samples is not None:
            auto_conditioning, diffusion_conditioning, auto_conds, _ = self.get_conditioning_latents(voice_samples,
                                                                                                     return_mels=True)
        elif conditioning_latents is not None:
            auto_conditioning, diffusion_conditioning = conditioning_latents
        else:
            auto_conditioning, diffusion_conditioning = self.get_random_conditioning_latents()
        auto_conditioning = auto_conditioning.to(self.device)
        diffusion_conditioning = diffusion_conditioning.to(self.device)

        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free,
                                                  cond_free_k=cond_free_k)

        with torch.no_grad():
            samples = []
            num_batches = num_autoregressive_samples // self.autoregressive_batch_size
            stop_mel_token = self.autoregressive.stop_mel_token
            calm_token = 83  # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output"
            if verbose:
                print("Generating autoregressive samples..")
            if not torch.backends.mps.is_available():
                with self.temporary_cuda(self.autoregressive
                                         ) as autoregressive, torch.autocast(device_type="cuda", dtype=torch.float16,
                                                                             enabled=self.half):
                    for b in tqdm(range(num_batches), disable=not verbose):
                        codes = autoregressive.inference_speech(auto_conditioning, text_tokens,
                                                                do_sample=True,
                                                                top_p=top_p,
                                                                temperature=temperature,
                                                                num_return_sequences=self.autoregressive_batch_size,
                                                                length_penalty=length_penalty,
                                                                repetition_penalty=repetition_penalty,
                                                                max_generate_length=max_mel_tokens,
                                                                **hf_generate_kwargs)
                        padding_needed = max_mel_tokens - codes.shape[1]
                        codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                        samples.append(codes)
            else:
                with self.temporary_cuda(self.autoregressive) as autoregressive:
                    for b in tqdm(range(num_batches), disable=not verbose):
                        codes = autoregressive.inference_speech(auto_conditioning, text_tokens,
                                                                do_sample=True,
                                                                top_p=top_p,
                                                                temperature=temperature,
                                                                num_return_sequences=self.autoregressive_batch_size,
                                                                length_penalty=length_penalty,
                                                                repetition_penalty=repetition_penalty,
                                                                max_generate_length=max_mel_tokens,
                                                                **hf_generate_kwargs)
                        padding_needed = max_mel_tokens - codes.shape[1]
                        codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                        samples.append(codes)

            clip_results = []

            if not torch.backends.mps.is_available():
                with self.temporary_cuda(self.clvp) as clvp, torch.autocast(
                        device_type="cuda" if not torch.backends.mps.is_available() else 'mps', dtype=torch.float16,
                        enabled=self.half
                ):
                    if cvvp_amount > 0:
                        if self.cvvp is None:
                            self.load_cvvp()
                        self.cvvp = self.cvvp.to(self.device)
                    if verbose:
                        if self.cvvp is None:
                            print("Computing best candidates using CLVP")
                        else:
                            print(
                                f"Computing best candidates using CLVP {((1 - cvvp_amount) * 100):2.0f}% and CVVP {(cvvp_amount * 100):2.0f}%")
                    for batch in tqdm(samples, disable=not verbose):
                        for i in range(batch.shape[0]):
                            batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                        if cvvp_amount != 1:
                            clvp_out = clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)
                        if auto_conds is not None and cvvp_amount > 0:
                            cvvp_accumulator = 0
                            for cl in range(auto_conds.shape[1]):
                                cvvp_accumulator = cvvp_accumulator + self.cvvp(
                                    auto_conds[:, cl].repeat(batch.shape[0], 1, 1), batch, return_loss=False)
                            cvvp = cvvp_accumulator / auto_conds.shape[1]
                            if cvvp_amount == 1:
                                clip_results.append(cvvp)
                            else:
                                clip_results.append(cvvp * cvvp_amount + clvp_out * (1 - cvvp_amount))
                        else:
                            clip_results.append(clvp_out)
                    clip_results = torch.cat(clip_results, dim=0)
                    samples = torch.cat(samples, dim=0)
                    best_results = samples[torch.topk(clip_results, k=k).indices]
            else:
                with self.temporary_cuda(self.clvp) as clvp:
                    if cvvp_amount > 0:
                        if self.cvvp is None:
                            self.load_cvvp()
                        self.cvvp = self.cvvp.to(self.device)
                    if verbose:
                        if self.cvvp is None:
                            print("Computing best candidates using CLVP")
                        else:
                            print(
                                f"Computing best candidates using CLVP {((1 - cvvp_amount) * 100):2.0f}% and CVVP {(cvvp_amount * 100):2.0f}%")
                    for batch in tqdm(samples, disable=not verbose):
                        for i in range(batch.shape[0]):
                            batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                        if cvvp_amount != 1:
                            clvp_out = clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)
                        if auto_conds is not None and cvvp_amount > 0:
                            cvvp_accumulator = 0
                            for cl in range(auto_conds.shape[1]):
                                cvvp_accumulator = cvvp_accumulator + self.cvvp(
                                    auto_conds[:, cl].repeat(batch.shape[0], 1, 1), batch, return_loss=False)
                            cvvp = cvvp_accumulator / auto_conds.shape[1]
                            if cvvp_amount == 1:
                                clip_results.append(cvvp)
                            else:
                                clip_results.append(cvvp * cvvp_amount + clvp_out * (1 - cvvp_amount))
                        else:
                            clip_results.append(clvp_out)
                    clip_results = torch.cat(clip_results, dim=0)
                    samples = torch.cat(samples, dim=0)
                    best_results = samples[torch.topk(clip_results, k=k).indices]
            if self.cvvp is not None:
                self.cvvp = self.cvvp.cpu()
            del samples

            # The diffusion model actually wants the last hidden layer from the autoregressive model as conditioning
            # inputs. Re-produce those for the top results. This could be made more efficient by storing all of these
            # results, but will increase memory usage.
            if not torch.backends.mps.is_available():
                with self.temporary_cuda(
                        self.autoregressive
                ) as autoregressive, torch.autocast(
                    device_type="cuda" if not torch.backends.mps.is_available() else 'mps', dtype=torch.float16,
                    enabled=self.half
                ):
                    best_latents = autoregressive(auto_conditioning.repeat(k, 1), text_tokens.repeat(k, 1),
                                                  torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                                                  best_results,
                                                  torch.tensor([best_results.shape[
                                                                    -1] * self.autoregressive.mel_length_compression],
                                                               device=text_tokens.device),
                                                  return_latent=True, generation_mode=True)
                    del auto_conditioning
            else:
                with self.temporary_cuda(
                        self.autoregressive
                ) as autoregressive:
                    best_latents = autoregressive(auto_conditioning.repeat(k, 1), text_tokens.repeat(k, 1),
                                                  torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                                                  best_results,
                                                  torch.tensor([best_results.shape[
                                                                    -1] * self.autoregressive.mel_length_compression],
                                                               device=text_tokens.device),
                                                  return_latent=True, generation_mode=True)
                    del auto_conditioning

            if verbose:
                print("Transforming autoregressive outputs into audio..")
            wav_candidates = []
            if not torch.backends.mps.is_available():
                with self.temporary_cuda(self.diffusion) as diffusion, self.temporary_cuda(
                        self.vocoder
                ) as vocoder:
                    for b in range(best_results.shape[0]):
                        codes = best_results[b].unsqueeze(0)
                        latents = best_latents[b].unsqueeze(0)

                        # Find the first occurrence of the "calm" token and trim the codes to that.
                        ctokens = 0
                        for k in range(codes.shape[-1]):
                            if codes[0, k] == calm_token:
                                ctokens += 1
                            else:
                                ctokens = 0
                            if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                                latents = latents[:, :k]
                                break
                        mel = do_spectrogram_diffusion(diffusion, diffuser, latents, diffusion_conditioning,
                                                       temperature=diffusion_temperature,
                                                       verbose=verbose)
                        wav = vocoder.inference(mel)
                        wav_candidates.append(wav.cpu())
            else:
                diffusion, vocoder = self.diffusion, self.vocoder
                diffusion_conditioning = diffusion_conditioning.cpu()
                for b in range(best_results.shape[0]):
                    codes = best_results[b].unsqueeze(0).cpu()
                    latents = best_latents[b].unsqueeze(0).cpu()

                    # Find the first occurrence of the "calm" token and trim the codes to that.
                    ctokens = 0
                    for k in range(codes.shape[-1]):
                        if codes[0, k] == calm_token:
                            ctokens += 1
                        else:
                            ctokens = 0
                        if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                            latents = latents[:, :k]
                            break
                    mel = do_spectrogram_diffusion(diffusion, diffuser, latents, diffusion_conditioning,
                                                   temperature=diffusion_temperature,
                                                   verbose=verbose)
                    wav = vocoder.inference(mel)
                    wav_candidates.append(wav.cpu())

            def potentially_redact(clip, text):
                if self.enable_redaction:
                    return self.aligner.redact(clip.squeeze(1), text).unsqueeze(1)
                return clip

            wav_candidates = [potentially_redact(wav_candidate, text) for wav_candidate in wav_candidates]

            if len(wav_candidates) > 1:
                res = wav_candidates
            else:
                res = wav_candidates[0]

            if return_deterministic_state:
                return res, (deterministic_seed, text, voice_samples, conditioning_latents)
            else:
                return res

    def deterministic_state(self, seed=None):
        """
        Sets the random seeds that tortoise uses to the current time() and returns that seed so results can be
        reproduced.
        """
        seed = int(time()) if seed is None else seed
        torch.manual_seed(seed)
        random.seed(seed)
        # Can't currently set this because of CUBLAS. TODO: potentially enable it if necessary.
        # torch.use_deterministic_algorithms(True)

        return seed


def save_latents_from_voices(voices_list, tts, out_dir='.'):
    conditioning_latents = tts.get_conditioning_latents(voices_list,
                                                        return_mels=False)

    if len(conditioning_latents) == 4:
        conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)

    outfile = f'{out_dir}/cond_latents_{tts.AR_model_home}.pth'
    torch.save(conditioning_latents, outfile)
    return conditioning_latents, outfile


def phonemizer(text, language="en-us", backend='espeak'):
    from phonemizer import phonemize
    from phonemizer.backend import BACKENDS

    def _get_backend(language="en-us", backend="espeak"):
        if backend == 'espeak':
            p = BACKENDS[backend](language, preserve_punctuation=True, with_stress=True)
        elif backend == 'espeak-mbrola':
            p = BACKENDS[backend](language)
        else:
            p = BACKENDS[backend](language, preserve_punctuation=True)

        return p

    if language == "en":
        language = "en-us"

    backend = _get_backend(language=language, backend=backend)
    if backend is not None:
        tokens = backend.phonemize([text], strip=True)
    else:
        tokens = phonemize([text], language=language, strip=True, preserve_punctuation=True, with_stress=True)

    return tokens[0] if len(tokens) == 0 else tokens


if __name__ == "__main__":
    import torchaudio
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model to load in.',
                        default='../training/lich_king/finetune/models/255_gpt.pth')  # '../base_models/autoregressive.pth'
    args = parser.parse_args()
    custom_model_path = args.model

    audio_dir = '../lich_king/audio'
    voice_samples = []
    for fn in os.listdir(audio_dir):
        voice_samples.append(load_audio(os.path.join(audio_dir, fn), 22050))

    c_t = CustomTTS(custom_autoregressive_path=custom_model_path,
                    models_dir='../base_models')
    conditioning_l, conditioning_f = save_latents_from_voices(voice_samples, c_t)

    test_text = "That's no moon."


    should_phonemize = True
    if should_phonemize:
        cut_text = phonemizer(test_text)

    gen = c_t.tts(test_text, voice_samples=None,
                  conditioning_latents=conditioning_l,
                  k=4,
                  verbose=False,
                  use_deterministic_seed=None,
                  return_deterministic_state=False,
                  # autoregressive generation parameters follow
                  num_autoregressive_samples=256,
                  temperature=0.6,
                  length_penalty=1,
                  repetition_penalty=2.0,
                  top_p=0.8,
                  max_mel_tokens=500,
                  cvvp_amount=0,
                  diffusion_iterations=200,
                  cond_free=True,
                  cond_free_k=2,
                  diffusion_temperature=1.0)
    for ind, sam in enumerate(gen):

        audio_ = sam.squeeze(0).cpu()
        torchaudio.save(os.path.join('./', f'test_{ind}.wav'), audio_, 24000)
