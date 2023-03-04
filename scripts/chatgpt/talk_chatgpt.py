from chatgpt_wrapper import ChatGPT
from text2audio import text_to_audio
from speech_rec import audio_to_text
import warnings
warnings.filterwarnings("ignore")

bot = ChatGPT()

for _ in range(4):
    my_ask = audio_to_text()
    response = bot.ask(my_ask)

    print(response)  # prints the response from chatGPT

    text_to_audio(response)  # converts the response to audio and plays it