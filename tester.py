from tokenizers import TokenizerV2, vocabs
# from data_prep import data_preparaion
import re

with open('the-verdict.txt', 'r') as file:
    raw_text = file.read()



# v2 = TokenizerV2(vocabs(raw_text))
# token = v2.encode(raw_text)
# detoken = v2.decode(token)
# print(detoken)

# data = data_preparaion.prepare(token, context_size=5)

# print(TokenizerV2(vocabs(raw_text)).encode(raw_text))

import os

# Get the number of CPU cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}", type(num_cores))
