from torch.utils.data import DataLoader, Subset
from dataset import StockImagesDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import StockCNN
import polars as pl
import os
from torch.utils.data import random_split

torch.manual_seed(42)

# ----- Parameters -----
BATCH_SIZE = 1024
NUM_WORKERS = 12
VALIDATION_SPLIT = .8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-5

# Early stopping parameters
PATIENCE = 2

# Train-validation split
SHUFFLE_IN_SAMPLE_DATASET = True
SHUFFLE_DATA_LOADER = False

# File paths
MODEL_NAME = f"{LEARNING_RATE}_1"
os.makedirs(f"weights/{MODEL_NAME}", exist_ok=True)

# Load dataset
in_sample_dataset = StockImagesDataset(
    annotations_file='data/train_annotations_20.csv',
    img_dir='/home/andrew/Data/images/20'
)



if SHUFFLE_IN_SAMPLE_DATASET:
    total_size = len(in_sample_dataset)
    val_size = int(VALIDATION_SPLIT * total_size)
    train_size = total_size - val_size
    train_dataset, validation_dataset = random_split(in_sample_dataset, [train_size, val_size])
else:
    split_index = int(VALIDATION_SPLIT * len(in_sample_dataset))
    train_dataset = Subset(in_sample_dataset, range(0, split_index))
    validation_dataset = Subset(in_sample_dataset, range(split_index, len(in_sample_dataset)))

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA_LOADER, num_workers=NUM_WORKERS)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA_LOADER, num_workers=NUM_WORKERS)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model
model = StockCNN().to(device)

# Loss function & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def validation() -> float:
    model.eval()
    val_losses = []
    with torch.no_grad():
        
        for images, labels in tqdm(validation_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels.float())
            val_losses.append(loss.item())

    return sum(val_losses) / len(val_losses)

# Training loop
best_val_loss = float('inf')
epochs_without_improvement = 0
early_stop = False
results = []
for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):

    if early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

    model.train()
    losses = []
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, labels.float())
        loss.backward()

        optimizer.step()
        
        losses.append(loss.item())
    
    train_loss = sum(losses) / len(losses)
    val_loss = validation()
    results.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'validation_loss': val_loss,
    })
    print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), f"weights/{MODEL_NAME}/{MODEL_NAME}_epoch_{epoch}.pth")

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), f"weights/{MODEL_NAME}/{MODEL_NAME}_epoch_best.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            early_stop = True

# Save Results
os.makedirs("results", exist_ok=True)
pl.from_dicts(results).write_parquet(f"results/{MODEL_NAME}_training_data.parquet")

