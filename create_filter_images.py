import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from model import StockCNN
from torch import nn
import polars as pl
from datetime import date
import os

os.makedirs("results/filters", exist_ok=True)
# Parameters
MODEL_NAME = "24_epochs_model_1"

ticker = "AAPL"
date_ = date(2019, 9, 30)

data = (
    pl.scan_parquet('data/crsp_daily.parquet')
    .filter(pl.col('ticker').eq(ticker))
    .filter(pl.col('date').eq(date_))
    .sort('date')
    .collect()
)

permno = data['permno'].last()
date_str = date_.strftime("%Y%m%d")
file_path = f"{permno}_{date_str}_20.png"

annotations = (
    pl.scan_csv('data/test_annotations_20.csv')
    .filter(pl.col('file').eq(file_path))
    .sort('file')
    .collect()
)

label = annotations['label'].last()

image_path = f"/home/andrew/Data/images/20/{permno}/{file_path}"
image = read_image(image_path)[:1].float()

# Save testing image
plt.imshow(image.squeeze(0), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.savefig("results/filters/original_image.png", dpi=300)
plt.clf()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = StockCNN()
model.load_state_dict(torch.load(f"weights/{MODEL_NAME}.pth"))
model.to(device)

# Evaluation
model.eval()

# Get convolutional layers
conv_layers = []
for child in model.children():
    for layer in child.children():
        if isinstance(layer, nn.Conv2d):
            conv_layers.append(layer)

# Put image on device and get results
image = image.to(device)
results = conv_layers[0](image)

# Plot a sample of the filters
plt.figure(figsize=(8, 5))
plt.suptitle("Convolution Filter Samples")

for i in range(4):
    filter = results[i]
    plt.subplot(2, 2, i + 1)
    plt.imshow(filter.detach().cpu().numpy())
    plt.axis('off')

plt.tight_layout()
plt.savefig("results/filters/filters.png", dpi=300)