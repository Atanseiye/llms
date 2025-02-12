import torch
from util import Softmax

def generate_text_sample(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        # Get the prediction
        with torch.no_grad():
            logits = model(idx_cond)

        # Take the last item alone
        logits = logits[:, -1, :]

        # Apply softmax to get probability
        probas = torch.softmax(logits, dim=-1)
        

        # Next Word
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append sampled index to the running index
        idx = torch.cat((idx, idx_next), dim=1)

    return idx