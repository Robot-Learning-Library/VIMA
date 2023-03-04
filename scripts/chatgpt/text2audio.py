import gtts
import os
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from pydub import effects

def text_to_audio(text):
    audio_path = "cache/test.mp3"
    # make request to google to get synthesis
    tts = gtts.gTTS(text)
    # save the audio file
    tts.save(audio_path)
    # load the audio file using pydub
    sound = AudioSegment.from_file(audio_path, format="mp3")
    sound = sound.speedup(1.3, 150, 25)
    # play the audio file
    play(sound)
    os.remove(audio_path)

if __name__ == "__main__":
    text_to_audio("Hello world")

