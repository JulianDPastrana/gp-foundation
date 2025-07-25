from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from chainedgp.datasets.toadstool import (
    ToadstoolSequentialDataset,
    stratified_split,
)
from chainedgp.rnn import MultiRateLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.set_default_dtype(torch.float64)


class MultiRateLSTMModel(torch.nn.Module):
    def __init__(self, input_size_list: list[int], hidden_size: int, output_size: int):
        super().__init__()
        self.multirate_lstm = MultiRateLSTM(
            input_size_list=input_size_list,
            hidden_size=hidden_size,
        )
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=False,
        )
        self.output_layer = torch.nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )

    def forward(
        self, inputs: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]
    ):
        # Pass through your multirate LSTM
        out, (ht, ct) = self.multirate_lstm(inputs)
        out, (ht, ct) = self.lstm(out)
        # Map to a single continuous output
        out = self.output_layer(ht[-1])
        return out


def toadstool_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Collate a batch where each sample is (x_dict, y),
    x_dict has keys "bvp", "eda", "hr", "acc" with shapes:
      - bvp: (T_bvp, 1)
      - eda: (T_eda, 1)
      - hr:  (T_hr, 1)
      - acc: (T_acc, 3)
    Returns:
      - x_list: [bvp_batch, eda_batch, hr_batch, acc_batch], where each
        tensor is shaped (sequence_length, batch_size, feature_dim)
      - y_batch: tensor shaped (batch_size, 1)
    """
    # unzip samples
    x_dicts, y_list = zip(*batch)

    # stack labels
    y_batch = torch.stack(y_list, dim=0)  # (B, 1)

    # Define the modality order you want in the list:
    modalities = ["bvp", "eda", "hr", "acc"]
    x_list: List[torch.Tensor] = []

    for mod in modalities:
        # gather and stack: result is (batch_size, seq_len, feat_dim)
        mod_batch = torch.stack([x_dict[mod] for x_dict in x_dicts], dim=0)
        # swap to (seq_len, batch_size, feat_dim)
        mod_batch = mod_batch.transpose(0, 1)
        x_list.append(mod_batch)

    return x_list, y_batch


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
        for x, y in tqdm(train_loader, desc="  Train", leave=False, unit="batch"):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train = running_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        for x, y in tqdm(valid_loader, desc="  Valid", leave=False, unit="batch"):
            with torch.no_grad():
                out = model(x)
                loss = loss_fn(out, y)
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


def compute_weights(subset):
    """
    Create a DataLoader with weighted sampling to approximate equal class frequency per batch.
    """
    # Gather labels from subset
    labels = [int(subset[i][1].item()) for i in range(len(subset))]
    counts = Counter(labels)
    sorted_counts = OrderedDict(sorted(counts.items()))
    weights = [len(subset) / counts_values for counts_values in sorted_counts.values()]
    return torch.tensor(weights)


def main():
    print(f"Using device: {DEVICE}")
    root = "~/Documents/data/toadstool-dataset/toadstool2/Toadstool 2.0"
    dataset = ToadstoolSequentialDataset(root, device=DEVICE)
    print(dataset)
    train_ds, valid_ds, test_ds = stratified_split(dataset, [0.9, 0.05, 0.05])
    print(f"Train: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")
    batch_size = 64

    # train_loader = make_balanced_loader(train_ds, batch_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=toadstool_collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=True, collate_fn=toadstool_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=toadstool_collate_fn
    )

    hidden_size = 64
    num_classes = len(dataset.labels)
    print(f"Hidden size: {hidden_size}, Number of classes: {num_classes}")

    # Set Up the model
    input_size_list = [6, 1, 4, 5]
    model = MultiRateLSTMModel(
        input_size_list=input_size_list,
        hidden_size=hidden_size,
        output_size=num_classes,
    )
    # Loss funciton

    weights = compute_weights(train_ds).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weights,
        reduction="mean",
        label_smoothing=0.0,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _ = train(
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
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            # Forward pass
            x = [xmod.cpu() for xmod in x]
            out = model(x)
            preds = out.argmax(dim=-1)
            # Accumulate
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
