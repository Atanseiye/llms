import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import YOR_GPT_CONFIG_124M
from v2_preprocessing import TokenizerV2, vocabs, create_dataloader_v1
from transformer import TransformerBlock
from layers import LayerNorm
import os

# ================================
# GPT Model Definition
# ================================
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emd = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emd = nn.Embedding(config['context_lenght'], config['emb_dim'])
        self.drop_emd = nn.Dropout(config['drop_rate'])

        # Transformer Blocks
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        # Layer Normalization & Output
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emd(in_idx)
        pos_embeds = self.pos_emd(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emd(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ================================
# Training Function
# ================================
def train_model(dataset_path, num_epochs=50, batch_size=32, learning_rate=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    tokenizer = TokenizerV2(vocabs(text_data))
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
    dataset_path = "the-verdict.txt"  # Change this to your dataset file
    train_model(dataset_path)
