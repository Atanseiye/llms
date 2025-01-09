import re
import tiktoken

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()


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


#            Version 1 Tokenizer
# ====================================================
class TokenizerV1:

    def __init__(self, vocabs):
        self.str_to_int = vocabs
        self.int_to_str = {i:s for s, i in vocabs.items()}

    def encode(self, text):
        splits = re.split(r'([.,:;\'"!_]|--|\s)', text)
        processed_tokens = [i.strip() for i in splits if i.strip()]
        ids = [self.str_to_int[s] for s in processed_tokens]
        return ids

    def decode(self, ids):
        text = ' '.join(self.int_to_str[s] for s in ids)
        text = re.sub(r'\s+([,.;:_!"\'()])', r'\1', text)
        return text



#            Version 2 Tokenizer
# =====================================================
class TokenizerV2:
    """
    This tokenizer uses bit-pair tokenization
    """

    def __init__(self, vocabs):
        self.str_to_int = vocabs
        self.int_to_str = {i:s for s,i in vocabs.items()}

    def encode(self, text):
        # text = vocabs(text)
        split = re.split(r'([,.\'"_!()]|--|\s)', text)
        processed_tokens = [i.strip() for i in split if i.strip()]
        processed_tokens = [
            item if item in self.str_to_int
            else '<|unk|>' for item in processed_tokens
        ]
        ids = [self.str_to_int[s] for s in processed_tokens]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[s] for s in ids])
        text = re.sub(r'\s+([,.;:_!"\'()])', r'\1', text)
        return text



# IMPLEMENTING BYTE PAIR TOKENIZATION USING tiktoken MODULE
# initialse the tiktoken libery
# tokenizer = tiktoken.get_encoding('gpt2')
# text = (
#     '''Hello, do you like tea? <|endoftext|> In the sunlit traces of the palace'''
# )
