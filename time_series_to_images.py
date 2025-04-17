import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from data_utils import load_yfinance
from indicators import transform, with_labels
import polars as pl
import os


def select_features(X, y, n_features=225):
    # Combine F-score and mutual information for feature selection, as per paper
    selector = SelectKBest(
        lambda X, y: f_classif(X, y)[0] + mutual_info_classif(X, y), k=n_features
    )
    selector.fit(X, y)
    return selector.get_feature_names_out()


def get_sampler(target):
    """
    Create a weighted random sampler to fix imbalanced classes.
    Source: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2

    TODO: Use [SMOTE](https://github.com/chingisooinar/SMOTE-Pytorch) instead
    """
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in target], dtype=np.float64)
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight, len(samples_weight))


def generate_datasets(data_path="data/wmt_data.parquet.gz"):
    if os.path.exists(data_path):
        df = pl.scan_parquet(data_path).collect()
    else:
        df = load_yfinance(["WMT"])
        df = with_labels(df)
        df = transform(df)
        df.write_parquet(data_path, compression="gzip")

    X = df.filter(pl.col("ticker") == "WMT").drop("ticker", "date", "label")
    label_map = {
        "BUY": 0,
        "SELL": 1,
        "HOLD": 2,
    }
    y = df["label"].replace(label_map).cast(pl.Int8)

    # Train and test split (sequential to avoid data leakage)
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    # Feature selection
    best_features = select_features(X_train, y_train)
    X_train = X_train[best_features]
    X_test = X_test[best_features]

    # Reshape to 2D images
    X_train = X_train.to_numpy().reshape(-1, 1, 15, 15)
    X_test = X_test.to_numpy().reshape(-1, 1, 15, 15)

    # Create datasets
    train = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.long),
    )
    test = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.long),
    )
    return train, test


def get_model(dropout=0.15):
    return nn.Sequential(
        # Conv 1
        # In: 15x15x1
        nn.Conv2d(1, 25, 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        # Conv 2
        # In: 14x14x25
        nn.Conv2d(25, 12, 2, stride=2),
        nn.ReLU(),
        nn.Dropout(dropout),
        # Linear 1
        # In: 7x7x12
        nn.Flatten(),
        nn.Linear(7 * 7 * 12, 100),
        nn.ReLU(),
        nn.Dropout(dropout),
        # Final Linear
        # In: 100
        nn.Linear(100, 3),
    )


def train(
    train, val, device, max_epochs=3000, bs=64, lr=0.001, warmup=0.1, **model_params
):
    train_loader = DataLoader(
        train,
        batch_size=bs,
        sampler=get_sampler(train.tensors[1].numpy()),
    )
    val_loader = DataLoader(val, batch_size=256, shuffle=False)

    model = get_model(**model_params).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer)

    train_losses = np.zeros(max_epochs)
    val_losses = np.zeros(max_epochs)
    for epoch in tqdm(range(max_epochs)):
        model.train()
        losses = []
        for X, y in train_loader:
            # Move to device
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for X, y in val_loader:
                # Move to device
                X = X.to(device)
                y = y.to(device)

                y_pred = model(X)
                val_loss += loss_fn(y_pred, y).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            train_losses[epoch] = np.mean(losses)
            val_losses[epoch] = val_loss
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: train loss {loss.item()}, val loss {val_loss}")

            if epoch > warmup * max_epochs and val_loss > np.mean(
                val_losses[epoch - 10 : epoch]
            ):
                print("Early stopping")
                break

    return model, train_losses[:epoch], val_losses[:epoch]


LOAD_MODEL = False
if __name__ == "__main__":
    # Generate datasets
    train_data, val_data = generate_datasets()

    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if LOAD_MODEL:
        model = get_model()
        model.load_state_dict(torch.load("model.pth"))
        model.to(device)
    else:
        # Train model
        model, train_losses, val_losses = train(train_data, val_data, device)

        # Save model
        torch.save(model.state_dict(), "model.pth")

        # Plot losses
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.show()

    # Evaluate
    model.eval()
    test_loader = DataLoader(val_data, batch_size=256, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y_true.append(y.numpy())
            y_pred.append(model(X).argmax(dim=1).cpu().numpy())
    # Flatten the lists
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    score = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced accuracy score: {score:.4f}")
    print(classification_report(y_true, y_pred, target_names=["BUY", "SELL", "HOLD"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


# .67 with 56 day normalization
