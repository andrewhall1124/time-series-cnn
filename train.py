from torch.utils.data import DataLoader, Subset
from dataset import StockImagesDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import StockCNN
import polars as pl

# ----- Parameters -----
BATCH_SIZE = 128
NUM_WORKERS = 12
VALIDATION_SPLIT = .8
NUM_EPOCHS = 1

# File paths
MODEL_NAME = "model_1"

# Load dataset
in_sample_dataset = StockImagesDataset(
    annotations_file='data/train_annotations_20.csv',
    img_dir='/home/andrew/Data/images/20'
)

split_index = int(VALIDATION_SPLIT * len(in_sample_dataset))
train_dataset = Subset(in_sample_dataset, range(0, split_index))
validation_dataset = Subset(in_sample_dataset, range(split_index, len(in_sample_dataset)))

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model
model = StockCNN().to(device)

# Loss function & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-8)

def validation() -> float:
    model.eval()
    val_losses = []
    with torch.no_grad():
        
        # step = 1
        for images, labels in tqdm(validation_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels.float())
            val_losses.append(loss.item())

            # if step > 100:
            #     break
            # else:
            #     step += 1

    return sum(val_losses) / len(val_losses)

# Training loop
results = []
for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
    model.train()
    losses = []
    
    # step = 1
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, labels.float())
        loss.backward()

        optimizer.step()
        
        losses.append(loss.item())

        # if step > 100:
        #     break
        # else:
        #     step += 1
    
    train_loss = sum(losses) / len(losses)
    val_loss = validation()
    results.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'validation_loss': val_loss,
    })
    print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save Results
pl.from_dicts(results).write_parquet(f"results/{MODEL_NAME}_training_data.parquet")
torch.save(model.state_dict(), f"weights/{MODEL_NAME}.pth")
