import torch
import torch.nn as nn
from config import YOR_GPT_CONFIG_124M
from preprocessing import TokenizerV2, vocabs
from trainer import GPTModel  # Import model from training script
import tiktoken

# ================================
# Load Model Function
# ================================
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}")

    # Initialize model
    model = GPTModel(YOR_GPT_CONFIG_124M).to(device)
    
    # Load saved weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()  # Set to evaluation mode
    return model, device


# ================================
# Generate Text Function
# ================================
def generate_text(model, tokenizer, seed_text, max_length=50):
    device = next(model.parameters()).device
    model.eval()

    # Tokenize input text
    input_ids = torch.tensor([tokenizer.encode(seed_text)], device=device)

    # Generate tokens iteratively
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)
        
        next_token_logits = logits[:, -1, :]  # Get last token predictions
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Select top token

        # Append to input sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)

        # Stop if end-of-text token is generated
        if next_token.item() == tokenizer.str_to_int.get("<|endoftext|>", -1):
            break

    # Decode tokens into text
    generated_text = tokenizer.decode(input_ids.squeeze(0).tolist())
    return generated_text


# ================================
# Run Inference
# ================================
if __name__ == "__main__":
    checkpoint_path = "yor_gpt_model.pth"  # Change this to your saved checkpoint
    seed_text = "It had always been his fate to have women say such things of him:"  # Example Yoruba input

    # Load model
    model, device = load_model(checkpoint_path)

    # Load tokenizer
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = TokenizerV2(vocabs(raw_text))

    # Generate text
    output = generate_text(model, tokenizer, seed_text)
    print("\nGenerated Text:\n", output)
