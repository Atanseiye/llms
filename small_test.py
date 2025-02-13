from GPT import GPTModel
from config import YOR_GPT_CONFIG_124M
from preprocessing import create_dataloader_v1
import tiktoken
import torch


# ================================
# Load Dataset
# ================================
with open('tokenizerss/cleaned_yoruba_text.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

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
    batch_size=1024,
    max_length= YOR_GPT_CONFIG_124M['context_lenght'], 
    stride= YOR_GPT_CONFIG_124M['context_lenght'], 
    num_workers=2,
    shuffle=True,
    drop_last=True
    )

val_loader = create_dataloader_v1(
    val_data, 
    batch_size=1024,
    max_length= YOR_GPT_CONFIG_124M['context_lenght'], 
    stride= YOR_GPT_CONFIG_124M['context_lenght'], 
    num_workers=2,
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
        if i >= num_batches:
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
    