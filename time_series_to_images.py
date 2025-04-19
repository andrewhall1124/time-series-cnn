import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import torch.nn.functional as F
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
from indicators import transform_and_save
import polars as pl
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

IM_DIM = 15
VAL_BS = 4096


def select_features(X, y):
    # Combine F-score and mutual information for feature selection, as per paper
    selector = SelectKBest(f_classif, k=IM_DIM**2)
    selector.fit(X, y)
    return selector.get_feature_names_out()


def generate_datasets(data_path="data/full.parquet.gz"):
    if os.path.exists(data_path):
        df = pl.read_parquet(data_path)
    else:
        df = load_daily_crsp(
            start_date=datetime.date(2009, 1, 1), end_date=datetime.date(2025, 3, 31)
        )
        df = transform_and_save(df)

    # Train/Test/Validation split by day to avoid data leakage
    dates = sorted(df["date"].unique().to_list())
    n_dates = len(dates)
    train_dates = dates[: int(n_dates * 0.8)]
    test_dates = dates[int(n_dates * 0.8) : int(n_dates * 0.9)]
    valid_dates = dates[int(n_dates * 0.9) :]

    df_train = df.filter(pl.col("date").is_in(train_dates))
    df_test = df.filter(pl.col("date").is_in(test_dates))
    df_valid = df.filter(pl.col("date").is_in(valid_dates))

    # TODO: compute covariance from last 100 days of training
    # from sklearn.covariance import LedoitWolf
    # Σ = LedoitWolf().fit(window).covariance_

    X_train = df_train.drop(["permno", "date", "label"])
    X_test = df_test.drop(["permno", "date", "label"])
    X_valid = df_valid.drop(["permno", "date", "label"])

    y_train = df_train["label"].cast(pl.Float64)
    y_test = df_test["label"].cast(pl.Float64)
    y_valid = df_valid["label"].cast(pl.Float64)

    # Backtesting will be performed on the test set (all data for each day is kept together)
    backtest_prices = df.select("permno", "date", "original_close").filter(
        pl.col("date").is_in(test_dates)
    )
    assert len(backtest_prices) == len(X_test)

    # Feature selection based on training data
    best_features = select_features(X_train, y_train)
    print("FEATURES", best_features)
    X_train = X_train[best_features]
    X_test = X_test[best_features]
    X_valid = X_valid[best_features]

    # Reshape to 2D images
    X_train = X_train.to_numpy().reshape(-1, 1, IM_DIM, IM_DIM)
    X_test = X_test.to_numpy().reshape(-1, 1, IM_DIM, IM_DIM)
    X_valid = X_valid.to_numpy().reshape(-1, 1, IM_DIM, IM_DIM)

    # Create datasets
    train = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32),
    )
    valid = torch.utils.data.TensorDataset(
        torch.tensor(X_valid, dtype=torch.float32),
        torch.tensor(y_valid.to_numpy(), dtype=torch.float32),
    )
    test = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.float32),
    )
    return train, valid, test, backtest_prices


class GaussianOutputLayer(nn.Module):
    def __init__(self, in_features):
        super(GaussianOutputLayer, self).__init__()
        self.mu = nn.Linear(in_features, 1)
        self.log_var = nn.Linear(in_features, 1)

    def forward(self, x):
        mu = self.mu(x) + 1
        log_var = self.log_var(x)
        var = torch.exp(log_var)
        return mu, var


def get_model(
    hidden_size=5,
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
        GaussianOutputLayer(hidden_size),
    )


def train(
    train,
    val,
    device,
    max_epochs=3000,
    bs=128,
    lr=5e-3,
    warmup=10,
    patience=5,
    **model_params,
):
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val, batch_size=VAL_BS, shuffle=False)

    model = get_model(**model_params).to(device)

    loss_fn = nn.GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer)

    train_losses = np.zeros(max_epochs)
    val_losses = np.zeros(max_epochs)
    best_model, best_val_loss, best_epoch = (None, None, None)
    for epoch in tqdm(range(max_epochs)):
        model.train()
        losses = []
        for X, y in train_loader:
            # Move to device
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            mu, var = model(X)
            loss = loss_fn(mu, y, var)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X, y in val_loader:
                # Move to device
                X = X.to(device)
                y = y.to(device)

                mu, var = model(X)
                val_loss += loss_fn(mu, y, var).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            train_loss = np.mean(losses)
            train_losses[epoch] = train_loss
            val_losses[epoch] = val_loss
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: train loss {train_loss}, val loss {val_loss}")

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(model.state_dict())
                best_epoch = epoch

            # Early stopping
            if epoch >= patience and epoch > warmup:
                # Check if validation loss hasn't improved for 'patience' epochs
                if val_loss > best_val_loss and epoch - best_epoch >= patience:
                    print(
                        f"Early stopping at epoch {epoch}. No improvement for {patience} epochs."
                    )
                    break
    # Load best model
    model.load_state_dict(best_model)

    return model, train_losses[:epoch], val_losses[:epoch]


def backtest(backtest_df, pred_mu, pred_var, initial_money=10_000, trading_days=251):
    """
    Simple backtest returning annualized return and Sharpe ratio
    (assumes zero risk‑free rate).
    """
    values = []
    backtest_df = backtest_df.with_columns(
        pl.Series(pred_mu).cast(pl.Float64).rename("pred_mu"),
        pl.Series(pred_var).cast(pl.Float64).rename("pred_var"),
    )

    money = initial_money
    allocations = {}
    for date in backtest_df["date"].unique().sort(descending=False):
        today = (
            backtest_df.filter(pl.col("date") == date)
            .with_columns(
                (pl.col("pred_mu") / pl.col("pred_var")).alias("weight_numerator")
            )
            .with_columns(
                (
                    pl.col("weight_numerator") / pl.col("weight_numerator").abs().sum()
                ).alias("weight")
            )
            .select(pl.col("permno"), pl.col("weight"), pl.col("original_close"))
        )

        # Close previous positions
        for permno, shares in allocations.items():
            price = today.filter(pl.col("permno") == permno)[
                "original_close"
            ].to_numpy()[0]
            # This could be a buy or sell depending on short vs long
            money += shares * price
        allocations = {}

        # Open new positions
        for permno, weight, price in today.iter_rows():
            weight = np.clip(weight, -1, 1)
            # Calculate number of shares to buy (note: allowing fractional shares)
            shares = money * weight / price
            allocations[permno] = shares
            # Update money
            money -= shares * price

        # Calculate portfolio value
        value = 0.0
        for permno, shares in allocations.items():
            price = today.filter(pl.col("permno") == permno)[
                "original_close"
            ].to_numpy()[0]
            value += shares * price

        values.append(value + money)

    values = np.array(values)

    # Plot value
    plt.plot(values)
    plt.title("Portfolio Value")
    plt.xlabel("Day")
    plt.ylabel("Value ($)")
    plt.show()

    return portfolio_stats(values, trading_days)


def portfolio_stats(values, trading_days=251):
    """
    Calculate portfolio statistics.
    """
    values = np.array(values)
    T = len(values)

    # Annualized return
    ann_return = (values[-1] / values[0]) ** (trading_days / T) - 1

    # Daily returns and Sharpe (zero RF)
    daily_rets = values[1:] / values[:-1] - 1
    sharpe = daily_rets.mean() / daily_rets.std(ddof=1) * np.sqrt(trading_days)

    return ann_return, sharpe


LOAD_MODEL = True
if __name__ == "__main__":
    # Generate datasets
    train_data, val_data, test_data, backtest_prices = generate_datasets()

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
    # Compute MSE and NLL
    test_loader = DataLoader(test_data, batch_size=VAL_BS, shuffle=False)
    pred_mu, pred_var, y_true = [], [], []
    test_nll, test_mse = 0.0, 0.0
    with torch.no_grad():
        for X, y in test_loader:
            # Move to device
            X = X.to(device)
            y = y.to(device)

            mu, var = model(X)
            mu = mu.squeeze()
            var = var.squeeze()

            # Save predictions
            pred_mu.append(mu.cpu().numpy())
            pred_var.append(var.cpu().numpy())
            y_true.append(y.cpu().numpy())

            # Compute NLL and MSE
            test_nll += F.gaussian_nll_loss(mu, y, var).item()
            test_mse += F.mse_loss(mu, y).item()

        test_nll /= len(test_loader)
        test_mse /= len(test_loader)
        print(f"Test NLL: {test_nll:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
    pred_mu = np.concatenate(pred_mu)
    pred_var = np.concatenate(pred_var)
    y_true = np.concatenate(y_true)

    ret, sharpe = backtest(backtest_prices, pred_mu, pred_var)
    print(f"Annualized return: {ret:.2%}")
    print(f"Sharpe ratio: {sharpe:.2f}")

    # Get buy baseline
    ret, sharpe = portfolio_stats(
        backtest_prices.group_by("date")
        .agg(pl.col("original_close").sum())
        .sort(pl.col("date"))
        .select("original_close")
        .to_numpy()
        .flatten()
    )
    print(f"Buy+Hold baseline annualized return: {ret:.2%}")
    print(f"Buy+Hold baseline Sharpe ratio: {sharpe:.2f}")
