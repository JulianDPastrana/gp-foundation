import torch
from torch.utils.data import DataLoader, Subset, random_split
from chainedgp.datasets.lorenz_attractor import LorenzAttractorDataset
from chainedgp.rnn import MultiRateLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Tuple
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1103
torch.set_default_dtype(torch.float64)


class MultiRateLSTMModel(torch.nn.Module):
    def __init__(self, input_size_list: list[int], hidden_size: int):
        super().__init__()
        # your custom multirate LSTM
        self.multirate_lstm = MultiRateLSTM(
            input_size_list=input_size_list,
            hidden_size=hidden_size,
        )
        # final regression head
        self.output_layer = torch.nn.Linear(
            in_features=hidden_size,
            out_features=1,
        )

    def forward(
        self, inputs: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]
    ):
        # Pass through your multirate LSTM
        _, (ht, _) = self.multirate_lstm(inputs)
        # Map to a single continuous output
        return self.output_layer(ht)


def lorenz_collate_fn(
    batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # batch is a list of ((x_i, y_i), z_i) tuples
    # unzip inputs and targets
    inputs, targets = zip(*batch)  # inputs: tuple of (x_i, y_i)
    xs, ys = zip(*inputs)  # xs: tuple of x_i, ys: tuple of y_i

    # stack into tensors of shape (batch_size, seq_len) or (batch_size,)
    x_batch = torch.stack(xs)  # -> (B, ...)
    y_batch = torch.stack(ys)  # -> (B, ...)

    # do your per-batch transform here:
    #  - swap batch and time dims
    #  - add a channel dim at the end
    x_batch = x_batch.transpose(0, 1).unsqueeze(-1)
    # if you also want to transform y_batch the same way, do it here:
    y_batch = y_batch.transpose(0, 1).unsqueeze(-1)

    z_batch = torch.stack(targets).unsqueeze(-1)  # -> (B, ...)

    return (x_batch, y_batch), z_batch


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = DEVICE,
):
    model.to(device)
    train_losses = []
    val_losses = []

    # outer loop with epoch progress bar
    for epoch in trange(1, epochs + 1, desc="Epoch", unit="ep"):
        # --- Training ---
        model.train()
        running_train_loss = 0.0

        # wrap the batch loop in tqdm for per‐batch progress
        for (x_batch, y_batch), z_batch in tqdm(
            train_loader, desc="  Train", leave=False, unit="batch"
        ):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch = z_batch.to(device)

            optimizer.zero_grad()
            out = model((x_batch, y_batch))
            loss = loss_fn(out, z_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train = running_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        for (x_batch, y_batch), z_batch in tqdm(
            valid_loader, desc="  Valid", leave=False, unit="batch"
        ):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch = z_batch.to(device)

            with torch.no_grad():
                out = model((x_batch, y_batch))
                loss = loss_fn(out, z_batch)
            running_val_loss += loss.item()

        avg_val = running_val_loss / len(valid_loader)
        val_losses.append(avg_val)

        # update the epoch bar with loss info
        tqdm.write(
            f"Epoch {epoch:3d}/{epochs} — "
            f"Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}"
        )

    # --- Plot training history ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"train": train_losses, "val": val_losses}


print(f"Using device: {DEVICE}")

# Create Datasets
dataset = LorenzAttractorDataset(num_samples=100_000, window_steps=2, dt=1e-3)
N = len(dataset)
print(f"Dataset length: {N}")

# compute split point
split_idx = int(0.8 * N)

# first 80% for training, last 20% for testing
train_indices = list(range(0, split_idx))
test_indices = list(range(split_idx, N))

train_ds, valid_ds = random_split(Subset(dataset, train_indices), [0.8, 0.2])
test_ds = Subset(dataset, test_indices)

print(f"Train: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")

batch_size = 10_000
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=lorenz_collate_fn
)
valid_loader = DataLoader(
    valid_ds, batch_size=batch_size, shuffle=True, collate_fn=lorenz_collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=lorenz_collate_fn
)

# Set Up the model
hidden_size = 15
input_size_list = [1, 2]
model = MultiRateLSTMModel(input_size_list=input_size_list, hidden_size=hidden_size)

# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training routine

history = train(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=10,
    device=DEVICE,
)
# --- Evaluation on test set ---
model.eval()
model.to("cpu")
z_true = []
z_pred = []
x_test = []
y_test = []

with torch.no_grad():
    for x, y in test_loader:
        # forward pass
        out = model(x)
        preds = out

        # move to CPU and store
        x_test.append(x[0].ravel().cpu())
        y_test.append(y[0].ravel().cpu())
        z_true.append(y.cpu())
        z_pred.append(preds.cpu())

x_test = torch.cat(x_test).numpy()
y_test = torch.cat(y_test).numpy()
z_true = torch.cat(z_true).numpy()
z_pred = torch.cat(z_pred).numpy()


mse = mean_squared_error(z_true, z_pred)
mae = mean_absolute_error(z_true, z_pred)
r2 = r2_score(z_true, z_pred)
print(f"Test MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

# --- 2) Grab the raw x and y test‐set signals from the Dataset using the same indices ---
# Assuming you still have `dataset` and `test_indices` from your split:
(rx, ry) = (1, 2)


# build “original-step” axes for each channel
# (i.e. how many integrator steps each sample corresponds to)
idx_z = np.arange(len(z_true))
idx_x = np.arange(len(x_test * rx))
idx_y = np.arange(len(y_test * ry))

# --- 3) Make two‐row figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# Upper: true vs. predicted z
ax1.scatter(idx_z, z_true, s=10, alpha=0.6, label="True z")
ax1.scatter(idx_z, z_pred, s=10, alpha=0.6, label="Predicted z")
ax1.set_ylabel("z value")
ax1.set_title("True vs. Predicted z on Test Set")
ax1.legend()

# Lower: x and y test signals (downsampled)
ax2.plot(idx_x, x_test, ".", alpha=0.6, label=f"x (stride={rx})")
ax2.plot(idx_y, y_test, ".", alpha=0.6, label=f"y (stride={ry})")
ax2.set_xlabel("Original Integrator Step Index")
ax2.set_ylabel("value")
ax2.set_title("Test x and y Signals (Downsampled)")
ax2.legend()

plt.tight_layout()
plt.show()
