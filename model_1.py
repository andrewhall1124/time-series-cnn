from torch.utils.data import DataLoader, Subset
from dataset import StockImagesDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(message)s")


# ----- Parameters -----
BATCH_SIZE = 2048
NUM_WORKERS = 12
VALIDATION_SPLIT = .8
NUM_EPOCHS = 100

# Load dataset
in_sample_dataset = StockImagesDataset(
    annotations_file='data/train_annotations_20.csv',
    img_dir='images/20'
)

out_of_sample_dataset = StockImagesDataset(
    annotations_file='data/test_annotations_20.csv',
    img_dir='images/20'
)

split_index = int(VALIDATION_SPLIT * len(in_sample_dataset))
train_dataset = Subset(in_sample_dataset, range(0, split_index))
validation_dataset = Subset(in_sample_dataset, range(split_index, len(in_sample_dataset)))

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(out_of_sample_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Define model
class StockCNN(nn.Module):
    def __init__(self):
        super(StockCNN, self).__init__()
        self.model = nn.Sequential()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # (batch, 1, 64, 60) → (batch, 32, 64, 60)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # (batch, 32, 64, 60) → (batch, 64, 64, 60)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample

        # Correcting the input size after conv2 + pooling:
        self.fc1 = nn.Linear(64 * 16 * 15, 128)  # Flattened features from conv2 + pooling layers
        self.fc2 = nn.Linear(128, 1)  # Single output for binary classification
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution + ReLU + Pooling
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor for the fully connected layers
        x = F.relu(self.fc1(x))  # Fully connected layer + ReLU
        x = self.dropout(x)  # Dropout layer for regularization
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x
    
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Instantiate the model
model = StockCNN().to(device)

# Loss function & Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
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
    logging.info(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

logging.info("Training complete!")


# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
logging.info(f"Test Accuracy: {accuracy:.2f}%")

# Save model weights
torch.save(model.state_dict(), "stock_cnn_weights.pth")
logging.info("Model weights saved.")

