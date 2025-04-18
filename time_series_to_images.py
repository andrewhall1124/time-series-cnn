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

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

IM_DIM = 15


def select_features(X, y):
    # Combine F-score and mutual information for feature selection, as per paper
    selector = SelectKBest(
        lambda X, y: f_classif(X, y)[0] + mutual_info_classif(X, y), k=IM_DIM**2
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

    # Save original close prices for backtesting
    test_prices = X_test["original_close"].to_numpy()
    X_train = X_train.drop("original_close")
    X_test = X_test.drop("original_close")

    # Feature selection
    best_features = select_features(X_train, y_train)
    print("FEATURES", best_features)
    X_train = X_train[best_features]
    X_test = X_test[best_features]

    # Reshape to 2D images
    X_train = X_train.to_numpy().reshape(-1, 1, IM_DIM, IM_DIM)
    X_test = X_test.to_numpy().reshape(-1, 1, IM_DIM, IM_DIM)

    # Create datasets
    train = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.long),
    )
    test = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.long),
    )
    return train, test, test_prices


def get_model(
    hidden_size=100,
    n_channels=(25, 12),
    activation=nn.ReLU,
):
    # After Conv1: (IM_DIM - 2 + 1) = IM_DIM - 1
    # After Conv2: floor((IM_DIM - 1 - 2) / 2 + 1) = (IM_DIM - 1) // 2
    final_dim = (IM_DIM - 1) // 2
    return nn.Sequential(
        # Conv 1
        # In: IM_DIM x IM_DIM x 1
        nn.Conv2d(1, n_channels[0], kernel_size=2),
        activation(),
        nn.BatchNorm2d(n_channels[0]),
        # Conv 2
        # In: (IM_DIM - 1) x (IM_DIM - 1) x n_channels[0]
        nn.Conv2d(n_channels[0], n_channels[1], kernel_size=2, stride=2),
        activation(),
        nn.BatchNorm2d(n_channels[1]),
        # Linear layers
        # In: final flattened features = n_channels[1] * final_dim * final_dim
        nn.Flatten(),
        nn.Linear(n_channels[1] * final_dim * final_dim, hidden_size),
        activation(),
        nn.BatchNorm1d(hidden_size),
        # Final output layer
        nn.Linear(hidden_size, 3),
    )


def train(
    train,
    val,
    device,
    max_epochs=3000,
    bs=128,
    lr=1e-4,
    warmup=0,
    patience=1,
    **model_params,
):
    train_loader = DataLoader(
        train,
        batch_size=bs,
        sampler=get_sampler(train.tensors[1].numpy()),
    )
    val_loader = DataLoader(val, batch_size=4096, shuffle=False)

    model = get_model(**model_params).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer)

    train_losses = np.zeros(max_epochs)
    val_losses = np.zeros(max_epochs)
    best_model, best_val_loss = (None, None)
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

            train_loss = np.mean(losses)
            train_losses[epoch] = train_loss
            val_losses[epoch] = val_loss
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: train loss {train_loss}, val loss {val_loss}")

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()

            # Early stopping
            if epoch >= patience and epoch > warmup:
                # Check if validation loss hasn't improved for 'patience' epochs
                if val_loss >= min(val_losses[max(0, epoch - patience) : epoch]):
                    print(
                        f"Early stopping at epoch {epoch}. No improvement for {patience} epochs."
                    )
                    break
    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses[:epoch], val_losses[:epoch]


def backtest(price_history, decisions, initial_money=10_000, trading_days=251):
    """
    Simple backtest returning annualized return and Sharpe ratio
    (assumes zero risk‑free rate).
    """
    money = initial_money
    shares = 0.0
    values = []

    for price, decision in zip(price_history, decisions):
        if decision == 0:  # Buy
            shares += money / price
            money = 0.0
        elif decision == 1:  # Sell
            money += shares * price
            shares = 0.0
        values.append(money + shares * price)

    values = np.array(values)
    T = len(values)

    # Plot value
    plt.plot(values)
    plt.title("Portfolio Value")
    plt.xlabel("Day")
    plt.ylabel("Value ($)")
    plt.show()

    # Annualized return
    ann_return = (values[-1] / initial_money) ** (trading_days / T) - 1

    # Daily returns and Sharpe (zero RF)
    daily_rets = values[1:] / values[:-1] - 1
    sharpe = daily_rets.mean() / daily_rets.std(ddof=1) * np.sqrt(trading_days)

    return ann_return, sharpe


LOAD_MODEL = False
if __name__ == "__main__":
    # Generate datasets
    train_data, val_data, backtest_prices = generate_datasets()

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

    ret, sharpe = backtest(backtest_prices, y_pred)
    print(f"Annualized return: {ret:.2%}")
    print(f"Sharpe ratio: {sharpe:.2f}")
