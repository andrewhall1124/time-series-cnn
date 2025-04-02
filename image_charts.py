import matplotlib.pyplot as plt
import polars as pl
from torchvision.io import read_image
from datetime import date
ticker = "AAPL"
date_ = date(2019, 9, 30)

data = (
    pl.scan_parquet('data/crsp_daily.parquet')
    .filter(pl.col('ticker').eq(ticker))
    .filter(pl.col('date').eq(date_))
    .sort('date')
    .collect()
)
print(data)

permno = data['permno'].last()
date_str = date_.strftime("%Y%m%d")
file_path = f"{permno}_{date_str}_20.png"

annotations = (
    pl.scan_csv('data/test_annotations_20.csv')
    .filter(pl.col('file').eq(file_path))
    .sort('file')
    .collect()
)
print(annotations)

label = annotations['label'].last()

image_path = f"/home/andrew/Data/images/20/{permno}/{file_path}"
image = read_image(image_path)[:1].float()

plt.title(f"Label: {label}")
plt.xticks([])
plt.yticks([])
plt.imshow(image.squeeze(0), cmap='gray')
plt.savefig(f"images/{permno}_{date_str}_20.png")



