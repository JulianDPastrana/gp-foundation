# transformer_ts_forecast.py
# Self-contained example: Transformer-encoder next-step training + multi-step forecasting
# Requirements: torch, matplotlib, numpy

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------
# Reproducibility & Device
# ----------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------
# Synthetic Dataset
# ----------------------
class NoisySineDataset(Dataset):
    """
    Each sample is a noisy sine wave segment with random frequency/phase/amplitude and optional trend.
    We return:
      x: first context_len points (shape [context_len, 1])
      y: the next true value (scalar target for next-step prediction)
      full_future: the next pred_len true values (only used for plotting/eval after training)
    """

    def __init__(self, n_samples=5000, context_len=64, pred_len=32):
        super().__init__()
        self.n = n_samples
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len

        self.data = []
        for _ in range(n_samples):
            freq = np.random.uniform(0.003, 0.015)  # cycles per step
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.5, 1.5)
            trend = np.random.uniform(-0.005, 0.005)  # small linear trend
            t = np.arange(self.total_len, dtype=np.float32)

            series = amp * np.sin(2 * np.pi * freq * t + phase) + trend * t
            noise = np.random.normal(scale=0.1, size=self.total_len).astype(np.float32)
            series = series + noise

            # normalize per-sample to zero mean, unit std (helps training)
            series = (series - series.mean()) / (series.std() + 1e-6)

            ctx = series[:context_len]
            one_ahead = series[context_len]  # single next-step label
            future = series[context_len : context_len + pred_len]
            self.data.append((ctx, one_ahead, future))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ctx, one_ahead, future = self.data[idx]
        # shape [T, 1] for model (feature dim = 1)
        x = torch.from_numpy(ctx).unsqueeze(-1)  # [context_len, 1]
        y = torch.tensor(one_ahead, dtype=torch.float32)  # scalar
        fut = torch.from_numpy(future).float()  # [pred_len]
        return x, y, fut


# ----------------------
# Positional Encoding (sine-cosine)
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # buffer => not a parameter, moves with .to(device)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x: [T, B, d_model]
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)  # broadcast over batch


# ----------------------
# Transformer Encoder Model
# ----------------------
class TransformerForecast(nn.Module):
    """
    Encoder-only model. Predicts the next value given a context window.
    Uses a causal mask inside the encoder to prevent peeking into the future within the window.
    """

    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        input_dim=1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),  # predict scalar next value
        )

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device):
        # standard causal mask: allow attending to current and previous, block future
        # shape [sz, sz] with -inf above diagonal
        mask = torch.full((sz, sz), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x):
        """
        x: [B, T, 1] -> we’ll permute to [T, B, 1]
        Returns: next-step prediction for each sequence in the batch, shape [B]
        """
        B, T, _ = x.shape
        x = self.input_proj(x)  # [B, T, d_model]
        x = x.permute(1, 0, 2)  # [T, B, d_model] for transformer
        x = self.pos_encoder(x)  # add positional enc
        src_mask = self.generate_causal_mask(T, x.device)  # [T, T]
        enc = self.encoder(x, mask=src_mask)  # [T, B, d_model]
        last_token = enc[-1]  # [B, d_model] (use final time step)
        out = self.head(last_token).squeeze(-1)  # [B]
        return out


# ----------------------
# Training Utilities
# ----------------------
def train_model(model, loader, val_loader, epochs=10, lr=1e-3):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, _ in loader:
            xb = xb.to(device)  # [B, T, 1]
            yb = yb.to(device)  # [B]
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        sched.step()

        print(
            f"Epoch {ep:02d}/{epochs} | train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f} | lr: {sched.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Load best params
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


# ----------------------
# Autoregressive Forecast
# ----------------------
@torch.no_grad()
def autoregressive_forecast(model, context, steps):
    """
    context: Tensor [T, 1] on CPU; we'll keep a rolling window
    steps: number of future steps to predict
    returns: [steps] predictions (CPU numpy)
    """
    model.eval()
    ctx = context.clone().to(device)  # [T, 1]
    preds = []
    for _ in range(steps):
        # model expects [B, T, 1]
        inp = ctx.unsqueeze(0)  # [1, T, 1]
        next_val = model(inp).squeeze(0)  # scalar
        preds.append(next_val.item())
        # append and slide window by 1
        ctx = torch.cat([ctx, next_val.view(1, 1)], dim=0)  # [T+1, 1]
        ctx = ctx[-context.size(0) :]  # keep last T
    return np.array(preds, dtype=np.float32)


# ----------------------
# Main
# ----------------------
def main():
    # Hyperparameters
    context_len = 250
    pred_len = 50
    n_train = 15000
    n_val = 1000
    batch_size = 512
    epochs = 15
    lr = 2e-3

    # Data
    full_train = NoisySineDataset(
        n_samples=n_train, context_len=context_len, pred_len=pred_len
    )
    full_val = NoisySineDataset(
        n_samples=n_val, context_len=context_len, pred_len=pred_len
    )

    train_loader = DataLoader(
        full_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        full_val, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # Model
    model = TransformerForecast(
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        input_dim=1,
    ).to(device)

    # Train
    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)

    # Pick a random validation sample to visualize
    x_ctx, _, y_future = full_val[np.random.randint(0, len(full_val))]
    # Autoregressive multi-step forecast
    y_pred = autoregressive_forecast(model, x_ctx, steps=len(y_future))

    # For plotting, also show the single-step prediction at t=context_len (first element of y_pred)
    ctx_np = x_ctx.squeeze(-1).numpy()  # [context_len]
    gt_np = y_future.numpy()  # [pred_len]
    pred_np = y_pred  # [pred_len]

    # ----------------------
    # Plot
    # ----------------------
    plt.figure(figsize=(10, 5))
    t_ctx = np.arange(len(ctx_np))
    t_future = np.arange(len(ctx_np), len(ctx_np) + len(gt_np))

    plt.plot(t_ctx, ctx_np, label="Context (observed)")
    plt.plot(t_future, gt_np, label="Ground truth future")
    plt.plot(t_future, pred_np, label="Forecast (Transformer)")
    plt.axvline(x=len(ctx_np) - 1, linestyle="--", alpha=0.5)  # split marker
    plt.title("Transformer Encoder: Next-step Training, Autoregressive Forecasting")
    plt.xlabel("Time step")
    plt.ylabel("Value (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
