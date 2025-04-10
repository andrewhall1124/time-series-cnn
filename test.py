from model import StockCNN
import torch
from tqdm import tqdm
from dataset import StockImagesDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import polars as pl

# ----- Parameters -----
BATCH_SIZE = 1024
NUM_WORKERS = 12

# File paths
MODEL_NAME = "100_0.001_epochs_model_1"

# Dataset
out_of_sample_dataset = StockImagesDataset(
    annotations_file='data/test_annotations_20.csv',
    img_dir='/home/andrew/Data/images/20'
)

# Dataloader
test_loader = DataLoader(out_of_sample_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = StockCNN()
model.load_state_dict(torch.load(f"weights/{MODEL_NAME}.pth"))
model.to(device)

# Evaluation
model.eval()

# ----- Probability Results -----
probabilities_list = []
with torch.no_grad():  # Disable gradient calculation
    for images, labels in tqdm(test_loader, desc="Testing Probabilities"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)

        probabilities = F.sigmoid(outputs)

        probabilities_list.append(probabilities)

results = torch.concat(probabilities_list).view(-1).tolist()
results = pl.Series("probability", results).to_frame()
print(results)

results.write_parquet(f"results/{MODEL_NAME}_test_results.parquet")
