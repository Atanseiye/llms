from GPT import GPTModel
from config import YOR_GPT_CONFIG_124M
from preprocessing import create_dataloader_v1
from postprocessing import generate_text_sample, generate_text_sample_, sample_from_top_k_p
from v2_preprocessing import text_to_token_ids, token_ids_to_text
import time
import tiktoken
import torch


dataset_path = "tokenizerss/cleaned_yoruba_text.txt"  # Change this to your dataset file

# ================================
# Load Dataset
# ================================
def load_data():
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    return text_data

text_data = load_data()
text_data = text_data[:30000]
# ================================
# Tokenize Dataset
# ================================
tokenizer = tiktoken.get_encoding('gpt2')
tokenizer.encode(text_data)

total_token = len(tokenizer.encode(text_data))

# ================================
# Create DataLoader - Training and Validation
# ================================
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(42)

train_loader = create_dataloader_v1(
    train_data, 
    batch_size=2,
    max_length= YOR_GPT_CONFIG_124M['context_lenght'], 
    stride= YOR_GPT_CONFIG_124M['context_lenght'], 
    num_workers=0,
    shuffle=True,
    drop_last=True
    )

val_loader = create_dataloader_v1(
    val_data, 
    batch_size=2,
    max_length= YOR_GPT_CONFIG_124M['context_lenght'], 
    stride= YOR_GPT_CONFIG_124M['context_lenght'], 
    num_workers=0,
    shuffle=False,
    drop_last=False
    )

# ================================
# Sanity Check
# ================================

if total_token * (train_ratio) < YOR_GPT_CONFIG_124M['context_lenght']:
    print('Data is too small for the model')
    print('Please increase the size of the dataset or reduce the context length')

if total_token * (1 - train_ratio) < YOR_GPT_CONFIG_124M['context_lenght']:
    print('Validation data is too small for the model')
    print('Please increase the size of the dataset or reduce the context length')

# ================================
# Print the shape of the train and validation data
# ================================
for x, y in train_loader:
    print('Train Dataset', x.shape, y.shape)
    

for x, y in val_loader:
    print('Test Dataset', x.shape, y.shape)
    

# ================================
# tokens
# ================================
print('')
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print('Train Tokens:', train_tokens)
print('Validation Tokens:', val_tokens)
print('Total Tokens:', train_tokens + val_tokens)

# ================================
# Model
# ================================
torch.manual_seed(42)
model = GPTModel(YOR_GPT_CONFIG_124M)
model.eval()

# ================================
# calculate loss
# ================================
def calc_loss_batch(input_data, target_data, model, device):
    input_data = input_data.to(device)
    target_data = target_data.to(device)

    logits = model(input_data)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_data.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float('nan')
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_data, target_data) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_data, target_data, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# ================================
# Device
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ================================
# Calculate Loss
# ================================
torch.manual_seed(42)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=1)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=1)

print('Train Loss:', train_loss)
print('Validation Loss:', val_loss)
    
# ================================
# Evaluate Model
# ================================
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ================================
# Generating and printing Samples
# ===============================_
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emd.weight.shape[0]
    encoded_text = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = sample_from_top_k_p(
            model=model, idx=encoded_text,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
        model.train()

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track loss and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation steps
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1}/{num_epochs} (Step {global_step:06d}):"
                      f" Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Generate and print sample
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

    # Return losses after all epochs
    return train_losses, val_losses, track_tokens_seen

    




# ================================
# Run Training
# ================================
if __name__ == "__main__":
    start_time = time.time()
    model = GPTModel(YOR_GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=.1)
    num_epochs = 20
    train_loss, val_loss, token_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Nígbà tó ṣe mí", tokenizer=tokenizer
    )
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f'Training Completed in {execution_time:.2f} minutes.')