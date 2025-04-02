import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt


MODEL_NAME = '10_epochs_model_1'

data = (
    pl.read_parquet(f"results/{MODEL_NAME}_training_data.parquet")
)

plt.figure(figsize=(10, 6))

sns.lineplot(data, x='epoch', y='train_loss', label='Training')
sns.lineplot(data, x='epoch', y='validation_loss', label='Validation')

plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig(f"results/{MODEL_NAME}_loss_chart.png")