import re
import tiktoken


#            Vocabullary Generator
# ====================================================
def vocabs(raw_text):
    tokens = re.split(r'([,.;:_\'!"()]|--|\s)', raw_text)

    processed_tokens = [i.strip() for i in tokens if i.strip()]
    # Creating token IDs
    unique_tokens = sorted(set(processed_tokens))
    unique_tokens.extend(['<|endoftext|>', '<|unk|>'])

    # the vocabulary of the token IDs
    vocabs = {token:integer for integer,token in enumerate(unique_tokens)}
    return vocabs






# IMPLEMENTING BYTE PAIR TOKENIZATION USING tiktoken MODULE
# initialse the tiktoken libery
# tokenizer = tiktoken.get_encoding('gpt2')
# text = (
#     '''Hello, do you like tea? <|endoftext|> In the sunlit traces of the palace'''
# )
