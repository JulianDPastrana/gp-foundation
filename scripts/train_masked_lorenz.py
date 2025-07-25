import torch
from torch.utils.data import DataLoader, Subset, random_split
from chainedgp.datasets.lorenz_attractor import LorenzAttractorDataset
import math

# from chainedgp.rnn import MultiRateLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Tuple
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1103
torch.set_default_dtype(torch.float64)


class MultiRateLSTMModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # your custom multirate LSTM
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
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
        _, (ht, _) = self.lstm(inputs)
        # Map to a single continuous output
        return self.output_layer(ht[-1])


def lorenz_collate_fn(
    batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    batch: list of ((x_i, y_i), z_i), where
      x_i: Tensor of shape (Lx,)
      y_i: Tensor of shape (Ly,)
      z_i: scalar Tensor

    Returns:
      x_aligned: Tensor of shape (T, B, 2), where
        T = lcm(Lx, Ly), B = batch size, 2 = [x,y] channels
        missing slots are filled with 0.
      z_batch:   Tensor of shape (B, 1)
    """
    # 1) unzip
    inputs, targets = zip(*batch)
    xs_seq, ys_seq = zip(*inputs)

    # 2) stack per‐modality into (Lx, B) and (Ly, B)
    x_stack = torch.cat(xs_seq, dim=-1)  # (Lx, B)
    y_stack = torch.cat(ys_seq, dim=-1)  # (Ly, B)
    # 3) compute LCM of their lengths
    Lx, B = x_stack.shape
    Ly, _ = y_stack.shape
    T = math.lcm(Lx, Ly)

    # 4) prepare empty aligned tensor
    device = x_stack.device
    dtype = x_stack.dtype
    x_aligned = torch.zeros((T, B, 2), device=device, dtype=dtype)

    # 5) compute the “stretch” ratios
    rx = T // Lx
    ry = T // Ly

    # 6) compute the time‐indices
    idx_x = torch.arange(Lx, device=device) * rx  # shape (Lx,)
    idx_y = torch.arange(Ly, device=device) * ry  # shape (Ly,)

    # 7) scatter into the aligned tensor
    x_aligned[idx_x, :, 0] = x_stack
    x_aligned[idx_y, :, 1] = y_stack

    # 8) stack targets into (B,1)
    z_batch = torch.stack(targets, dim=0)

    return x_aligned, z_batch


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
        for inputs_train, targets_train in tqdm(
            train_loader, desc="  Train", leave=False, unit="batch"
        ):
            optimizer.zero_grad()
            out = model(inputs_train)
            loss = loss_fn(out, targets_train)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train = running_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        for inputs_valid, targets_valid in tqdm(
            valid_loader, desc="  Valid", leave=False, unit="batch"
        ):
            with torch.no_grad():
                out = model(inputs_valid)
                loss = loss_fn(out, targets_valid)
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
dataset = LorenzAttractorDataset(
    num_samples=50_000, window_steps=2, dt=1e-2, device=DEVICE
)
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

batch_size = 64
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lorenz_collate_fn,
    # pin_memory=True,
    # num_workers=1,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lorenz_collate_fn,
    # pin_memory=True,
    # num_workers=1,
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=lorenz_collate_fn
)

# Set Up the model
hidden_size = 15
input_size = 2
model = MultiRateLSTMModel(input_size=input_size, hidden_size=hidden_size)

# Loss function
loss_fn = torch.nn.MSELoss(reduction="mean")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training routine

history = train(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=100,
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
        print(x.shape, y.shape)
        # forward pass
        out = model(x.cpu())
        preds = out

        # move to CPU and store
        x_test.append(x[0].ravel().cpu())
        y_test.append(x[1].ravel().cpu())
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
idx_x = np.arange(len(x_test))
idx_y = np.arange(len(y_test))

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
