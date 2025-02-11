from preprocessing import create_dataloader_v1
import torch
import os

# ==================
print(torch.__version__)
if __name__ == "__main__":
    dataloaders = create_dataloader_v1(
        raw_text, batch_size=1, max_length=10, stride=2, shuffle=False, num_workers=os.cpu_count()
    )

    data_iter = iter(dataloaders)
    first_batch = next(data_iter)
    print(first_batch)