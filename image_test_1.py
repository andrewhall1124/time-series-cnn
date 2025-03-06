import data_utils as du
from indicators import transform
import polars as pl
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from datetime import date

# Load and transform data
df = du.load_yfinance(['WMT'])
df = transform(df)

# Select the last 225 rows and last 225 columns
df = (
    df
    .filter(pl.col('date').eq(date(2024, 11, 27)))
    .unpivot(index=['date', 'ticker'])
    .with_columns(
        pl.col('variable').str.splitn("_", 2).struct.rename_fields(['indicator', 'lag']).alias('fields')
    )
    .unnest('fields')
    .select(['indicator', 'lag', 'value'])
)

df = (
    df
    .pivot(on='indicator', index='lag', values='value')
    .slice(-13, 13)
)

print(df)

# # Convert Polars DataFrame to NumPy array
data = df.drop('lag').to_numpy()

# Normalize the data between 0 and 1 (optional, but recommended)
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min + 1e-8)  # Avoid division by zero

# # Convert NumPy array to PIL Image (for compatibility with torchvision)
image = Image.fromarray((data * 255).astype(np.uint8))  # Convert to 8-bit grayscale

# Convert PIL Image to PyTorch Tensor
transform = transforms.ToTensor()  # Converts to shape (C, H, W)
tensor_image = transform(image)  # Now a PyTorch tensor

# Print the shape to verify (Should be: (1, 225, 225))
print("Tensor shape:", tensor_image.shape)

# Example: Save the tensor as an image (Optional)
image.save("test_image_1.png")
