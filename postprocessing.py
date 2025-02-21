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

def sample_from_top_k_p(probas, k=50, p=0.9, temperature=1.0):
    """
    Applies top-k and top-p (nucleus) sampling with temperature scaling.
    """
    
    # Apply temperature scaling
    probas = probas ** (1.0 / temperature)
    probas = probas / probas.sum(dim=-1, keepdim=True)
    
    # Get top-k probabilities and indices
    top_k_values, top_k_indices = torch.topk(probas, k)
    
    # Apply top-p (nucleus) filtering
    cumulative_probs = torch.cumsum(top_k_values, dim=-1)
    mask = cumulative_probs > p
    mask[:, 1:] = mask[:, :-1].clone()  # Shift mask right to retain first valid token
    mask[:, 0] = False  # Always keep at least one token
    top_k_values[mask] = 0  # Mask out low probability tokens
    
    # Normalize after filtering
    top_k_values /= top_k_values.sum(dim=-1, keepdim=True)
    
    # Sample from the filtered distribution
    idx_next = torch.multinomial(top_k_values, 1)
    return torch.gather(top_k_indices, -1, idx_next)


def generate_text_sample_(model, idx, max_new_tokens, context_size, k=50, p=0.9, temperature=1.0):
    """
    Generates text using top-k, top-p sampling and temperature scaling.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        # Get the prediction
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Take the last timestep
        logits = logits[:, -1, :]
        
        # Apply softmax to get probability distribution
        probas = torch.softmax(logits, dim=-1)
        
        # Sample from the top-k, top-p distribution
        idx_next = sample_from_top_k_p(probas, k=k, p=p, temperature=temperature)
        
        # Append sampled index to the running index
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx