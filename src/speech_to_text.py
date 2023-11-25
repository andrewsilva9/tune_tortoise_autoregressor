"""
Created by Andrew Silva on 11/25/23

Demo of using whisper to run STT (for transcriptions or local interaction)
"""

import whisper

model = whisper.load_model('base')
result = model.transcribe('../lich_king/audio/LK Annoyed2.wav')
print(result["text"])
