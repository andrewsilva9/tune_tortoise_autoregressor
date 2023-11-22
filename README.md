# tune_tortoise_autoregressor

The goal of this repository is to provide a fairly self-contained set of scripts for fine-tuning the TorToiSe text-to-speech system.
There is an existing web-app that you can use if you just want easy/accessible fine-tuning and inference: https://git.ecker.tech/mrq/ai-voice-cloning -- the purpose of this repo is to provide a self-contained fine-tuning setup for developers.


The original TorToiSe TTS Author (James Betker) has produced a nice write-up of how the system works here: https://arxiv.org/pdf/2305.07243.pdf

### Where is this code from?
Much of this code is copied from or trimmed out of DLAS: https://github.com/neonbjb/DL-Art-School -- I've tried to basically just extract the necessary components.

I also used a lot of this demo app: https://git.ecker.tech/mrq/ai-voice-cloning to pull out necessary training pieces.
No actual code was copied from the gradio app, but it was immensely useful in recreating the training process with DLAS and seeing which relevant bits I needed to extract.
While the linked app does everything I have reproduced here (and more), I wanted the code to be accessible outside of a gradio web-app.

### Requirements

This requires Python 3.10 and requirements in the requirements.txt file.

You also need the base models (from which to fine-tune), which are available in this google drive folder:
https://drive.google.com/drive/folders/1o1BQ6wlRonQu0FAi8SdQZMaNeDqxVm5I?usp=sharing

Download them to the `base_models` directory. I'd include them here, but I don't want to pay for git LFS so...

### Fine-Tuning:
I've included some sample files in the `lich_king` directory to get you started. You'll need :

* a training YAML (see `lich_king/train.yaml` for an example) that links to your training and validation data.
* a `.txt` of training data, with paths to audio files and transcriptions (see `lich_king/train.txt`)
* a directory of audio files (preferably `.wav`) (see `lich_king/audio/*.wav`)
* (Optionally) a `.txt` of validation data and validation audio files, though these aren't super necessary or useful.

With that, navigate to the `src` directory and run
```bash
python train.py --yaml ../lich_king/train.yaml
```
And you'll see the model begin to train. Note that you can point `--yaml` to a different training yaml for a different fine tuned model.

Models will be saved into the `../training/{voice_name}/finetune/models/` directory.

### Generation with a fine-tuned model:
I've put a demo for generation inside `custom_tts.py`.

For that, run:

```bash
python custom_tts.py --model ../training/lich_king/finetune/models/255_gpt.pth
```
The model will generate the conditioning latents for the voices you've provided. The text and voices are currently hard coded (to be fixed soon!).

`--model` points to the model you want to load in.

The 4 versions of the speech will be saved as `test_k.wav` in the `src` directory. 