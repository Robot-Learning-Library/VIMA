import os
import openai

OPENAI_API_KEY = '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'
openai.api_key = f"{OPENAI_API_KEY}"

# AEGMRTV
# prompts = ["Q: There are five letters A, E, G, T, R. How to make them in order G, R, E, A, T? A: Put R on the right of G. Put E on the right of R. Put A on the right of E. Put T on the right of A. There are four letters E, A, R, T. How to make them in order R, A, T, E? A: Put A on the right of R. Put T on the right of A. Put E on the right of T. Q: There are four letters E, A, M, T. How to make them in order M, E, T, A? "]
prompts = [
        # "Q: There are three letters A, E, G. \
        # How to put them in order G, A, E? \
        # A: Put A on the right of G. \
        # Put E on the right of A. \
        # Q: There are three letters R, T, A. \
        # How to put them in order T, A, R? ", 

        # "Q: There are three letters A, E, G. \
        # How to put them in order G, A, E? \
        # A: Put A on the right of G. \
        # Put E on the right of A. \
        # Q: There are three letters A, E, G. \
        # How to put them in order G, E, A? \
        # A: Put E on the right of G. \
        # Put A on the right of E. \
        # Q: There are three letters A, E, G. \
        # How to put them in order A, G, E? \
        # A: Put G on the right of A. \
        # Put E on the right of G. \
        # Q: There are three letters M, V, R. \
        # How to put them in order R, V, M? ",

        # "Q: There are three letters A, E, G. \
        # How to put them in order G, A, E? \
        # A: Put A on the right of G. \
        # Put E on the right of A. \
        # Q: There are three letters A, E, G. \
        # How to put them in order G, E, A? \
        # A: Put E on the right of G. \
        # Put A on the right of E. \
        # Q: There are three letters A, E, G. \
        # How to put them in order A, G, E? \
        # A: Put G on the right of A. \
        # Put E on the right of G. \
        # Q: There are three letters M, V, R. \
        # How to put them in order R, M, V? ",

        # "Q: There are three letters A, E, G. \
        # How to put them in order E, A, G? \
        # A: Put A on the right of E. \
        # Put G on the right of A. \
        # Q: There are three letters A, E, G. \
        # How to put them in order E, G, A? \
        # A: Put G on the right of E. \
        # Put A on the right of G. \
        # Q: There are three letters M, V, R. \
        # How to put them in order M, R, V? ",

        # "Q: There are three letters A, E, G. \
        # How to put them in order E, A, G? \
        # A: Put A on the right of E. \
        # Put G on the right of A. \
        # Q: There are three letters A, E, G. \
        # How to put them in order E, G, A? \
        # A: Put G on the right of E. \
        # Put A on the right of G. \
        # Q: There are three letters A, E, G. \
        # How to put them in order A, G, E? \
        # A: Put G on the right of A. \
        # Put E on the right of G. \
        # Q: There are three letters A, E, G. \
        # How to put them in order A, E, G? \
        # A: Put E on the right of A. \
        # Put G on the right of E. \
        # Q: There are three letters A, E, G. \
        # How to put them in order G, A, E? \
        # A: Put A on the right of G. \
        # Put E on the right of A. \
        # Q: There are three letters A, E, G. \
        # How to put them in order G, E, A? \
        # A: Put E on the right of G. \
        # Put A on the right of E. \
        # Q: There are four letters M, V, R, A. \
        # How to put them in order A, M, R, V? ",

        "Q: There are three letters A, E, G. \
        How to put them in order G, A, E? \
        A: Put A on the right of G. \
        Put E on the right of A. \
        Q: There are four letters A, E, G, R. \
        How to put them in order G, A, R, E? \
        A: Put A on the right of G. \
        Put R on the right of A. \
        Put E on the right of R. \
        Q: There are four letters A, E, G, R. \
        How to put them in order A, G, E, R? \
        A: Put G on the right of A. \
        Put E on the right of G. \
        Put R on the right of E. \
        Q: There are four letters R, T, A, M. \
        How to put them in order R, T, A, M? ",

        "Q: There are three letters A, E, G. \
        How to put them in order G, A, E? \
        A: Put A on the right of G. \
        Put E on the right of A. \
        Q: There are four letters A, E, G, R. \
        How to put them in order G, A, R, E? \
        A: Put A on the right of G. \
        Put R on the right of A. \
        Put E on the right of R. \
        Q: There are four letters A, E, G, R. \
        How to put them in order A, G, E, R? \
        A: Put G on the right of A. \
        Put E on the right of G. \
        Put R on the right of E. \
        Q: There are four letters A, E, G, R. \
        How to put them in order E, A, R, G? \
        A: Put A on the right of E. \
        Put R on the right of A. \
        Put G on the right of R. \
        Q: There are four letters A, E, G, R. \
        How to put them in order E, R, A, G? \
        A: Put R on the right of E. \
        Put A on the right of R. \
        Put G on the right of A. \
        Q: There are four letters R, T, A, M. \
        How to put them in order R, T, A, M? "
        ]

for gpt_prompt in prompts:
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    print('----------------------\n', gpt_prompt, response['choices'][0]['text'])
