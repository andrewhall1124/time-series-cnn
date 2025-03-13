import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def get_model(dropout=0.15):
    return nn.Sequential(
        # Conv 1
        # In: 15x15x3
        nn.Conv2d(3, 25, 2),
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


def train(train, val, epochs=3000, bs=64, lr=0.001, warmup=0.1, **model_params):
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False)

    model = get_model(model_params)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer)

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    for epoch in tqdm(range(epochs)):
        model.train()
        losses = []
        for X, y in train_loader:
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
                y_pred = model(X)
                val_loss += loss_fn(y_pred, y).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            train_losses[epoch] = np.mean(losses)
            val_losses[epoch] = val_loss
            print(f"Epoch {epoch}: train loss {loss.item()}, val loss {val_loss}")

            if epoch > warmup * epochs and val_loss > np.mean(
                val_losses[epoch - 10 : epoch]
            ):
                print("Early stopping")
                break

    return model, train_losses, val_losses
