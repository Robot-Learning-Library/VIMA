import speech_recognition as sr
from pydub import AudioSegment
import os

# Initialize recognizer
r = sr.Recognizer()

# Record Audio
# with sr.Microphone() as source:
#     print("Speak something...")
#     audio = r.listen(source)

# load audio file
mp3_file = 'cache/test.mp3'
wav_file = 'cache/test.wav'
audio = AudioSegment.from_mp3(mp3_file)
audio.export(wav_file, format="wav")

with sr.AudioFile(wav_file) as source:
    audio = r.record(source)
os.remove(wav_file)

# Speech recognition using Google Speech Recognition
try:
    text = r.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))