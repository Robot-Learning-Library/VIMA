""" This is the OpenAI official interface for GPT-3.5-turbo-0301."""
import os
import openai

OPENAI_API_KEY = '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'
openai.api_key = f"{OPENAI_API_KEY}"

# AEGMRTV
asks = ["Hello, world!"]

for ask in asks:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": ask}])

        print('----------------------\n', ask, completion.choices[0].message.content)

