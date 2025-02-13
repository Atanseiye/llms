import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import YOR_GPT_CONFIG_124M
from v2_preprocessing import TokenizerV2, vocabs, create_dataloader_v1
from transformer import TransformerBlock
from layers import LayerNorm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from GPT import GPTModel
import os


# ================================
# Training Function
# ================================
def train_model(dataset_path, num_epochs=2, batch_size=256, learning_rate=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    dataloader = create_dataloader_v1(text_data, batch_size=batch_size, num_workers=3)

    # Initialize Model
    model = GPTModel(YOR_GPT_CONFIG_124M).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            # Reshape for cross-entropy loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete - Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


# ================================
# Run Training
# ================================
if __name__ == "__main__":
    dataset_path = "tokenizerss/yoruba_text.txt"  # Change this to your dataset file
    train_model(dataset_path)
