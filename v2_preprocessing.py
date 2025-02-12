import re
import torch
from torch.utils.data import DataLoader, Dataset

# ================================
# Dataset Class (Sliding Window)
# ================================
class DatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

# ================================
# DataLoader Function
# ================================
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = TokenizerV2(vocabs(txt))
    dataset = DatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# ================================
# TokenizerV1 (Basic)
# ================================
class TokenizerV1:
    def __init__(self, vocabs):
        self.str_to_int = vocabs
        self.int_to_str = {i: s for s, i in vocabs.items()}

    def encode(self, text):
        splits = re.split(r'([.,:;\'"!_]|--|\s)', text)
        processed_tokens = [i.strip() for i in splits if i.strip()]
        return [self.str_to_int.get(s, self.str_to_int['<|unk|>']) for s in processed_tokens]

    def decode(self, ids):
        text = ' '.join(self.int_to_str[s] for s in ids)
        return re.sub(r'\s+([,.;:_!"\'()])', r'\1', text)

# ================================
# TokenizerV2 (Improved with Fallback Encoding)
# ================================
class TokenizerV2:
    def __init__(self, vocabs):
        self.str_to_int = vocabs
        self.int_to_str = {i: s for s, i in vocabs.items()}

    def encode(self, text):
        split = re.split(r'([,.\'"_!()]|--|\s)', text)
        processed_tokens = [i.strip() for i in split if i.strip()]

        ids = []
        for token in processed_tokens:
            if token in self.str_to_int:
                ids.append(self.str_to_int[token])
            else:
                ids.extend([self.str_to_int.get(c, self.str_to_int['<|unk|>']) for c in token])

        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[s] for s in ids])
        return re.sub(r'\s+([,.;:_!"\'()])', r'\1', text)

# ================================
# Vocabulary Builder
# ================================
def vocabs(raw_text):
    tokens = re.split(r'([,.;:_\'!"()]|--|\s)', raw_text)
    processed_tokens = [i.strip() for i in tokens if i.strip()]
    unique_tokens = sorted(set(processed_tokens))
    unique_tokens.extend(['<|endoftext|>', '<|unk|>'])
    return {token: integer for integer, token in enumerate(unique_tokens)}

# ================================
# Embedding Class
# ================================
class embedding:
    @staticmethod
    def token_embedding_layer(vocab_size, output_dim):
        torch.manual_seed(123)
        return torch.nn.Embedding(vocab_size, output_dim)
