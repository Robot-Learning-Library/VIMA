import replicate
from os.path import exists
from config import config
import urllib.request
from pydub import AudioSegment
from functools import reduce

def normalize_audio_lufs(self, sound, target_lufs=-23.0):
    loudness_before = sound.dBFS
    sound = sound.apply_gain(target_lufs - loudness_before)
    return sound

def generate_audio_file():
    """Generate an audio file with the podcast conversation. File is saved in the output folder.
    Args:
        self.book_name (str): The name of the book.
        self.output_dir (str): The folder to save the output file.
    Returns:
        None. Writes an audio file in the output folder under the file name: self.output_dir / self.book_name + "_podcast.mp3"
    """
    model = "afiaka87/tortoise-tts"
    version = "e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71"
    client = replicate.Client(api_token=config.replicate_api_key)
    model = client.models.get(model)
    version = model.versions.get(version)

    sounds = []
    with open("test.txt", "r") as f:
        for ii, line in enumerate(f):
            print(line)
            voice = "daniel" if line[5] == "1" else "william"
            preset = "standard"
            output_fname = f"audio.mp3"
            if not exists(output_fname):

                output = version.predict(text=line, voice_a=voice, 
                                            reset=preset, cvvp_amount=1.0, seed=26031987)

                urllib.request.urlretrieve(output, output_fname)
            sound = normalize_audio_lufs(AudioSegment.from_file(output_fname, format="mp3"))
            sounds.append(sound)

    # sound1, with sound2 appended
    combined = reduce(lambda x, y: x+y, sounds)

    # export
    combined.export("podcast.mp3", format="mp3")


if __name__ == "__main__":
    generate_audio_file()