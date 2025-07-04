import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict

from chainedgp.datasets.toadstool import (
    ToadstoolSequentialDataset,
    stratified_split,
    make_balanced_loader,
)
from chainedgp.utils.training import train_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.set_default_dtype(torch.float64)


class MultiCellLSTM(torch.nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Lets's list all possible scenarios
        # 1. BVP (64 Hz) 1 feature
        # 2. BVP + ACC (32 Hz) 4 features
        # 3. BVP + ACC + EDA (4 Hz) 5 features
        # 4. BVP + ACC + EDA + HR (1 Hz) 6 features

        # One LSTM cell for each scenario
        self.cell1 = torch.nn.LSTMCell(input_size=1, hidden_size=hidden_size)
        self.cell2 = torch.nn.LSTMCell(input_size=4, hidden_size=hidden_size)
        self.cell3 = torch.nn.LSTMCell(input_size=5, hidden_size=hidden_size)
        self.cell4 = torch.nn.LSTMCell(input_size=6, hidden_size=hidden_size)

        self.cells_dict = {
            1: self.cell1,
            4: self.cell2,
            5: self.cell3,
            6: self.cell4,
        }

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x["bvp"]: (Tbvp (256), batch_size, 1)
        x["acc"]: (Tacc (128), batch_size, 3)
        x["eda"]: (Teda (16), batch_size, 1)
        x["hr"]:  (Thr (4), batch_size, 1)
        """
        x_bvp = x["bvp"]  # (Tbvp, batch_size, 1)
        x_acc = x["acc"]  # (Tacc, batch_size, 3)
        x_eda = x["eda"]  # (Teda, batch_size, 1)
        x_hr = x["hr"]  # (Thr, batch_size, 1)

        # Apply Batch normalization

        Tbvp, batch_size, _ = x_bvp.shape
        Tacc, _, _ = x_acc.shape
        Teda, _, _ = x_eda.shape
        Thr, _, _ = x_hr.shape

        # Compute the up‐sampling ratios to aling wiht the BVP signal
        ratio_acc = Tbvp // Tacc  # 256 // 128 == 2
        ratio_eda = Tbvp // Teda  # 256 // 16 == 16
        ratio_hr = Tbvp // Thr  # 256 // 4 == 64

        # Initialize the chared hidden state and cell state
        hx = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        cx = torch.zeros(batch_size, self.hidden_size, device=DEVICE)

        output = []
        hn, cn = [hx], [cx]
        # Let's loop over sequences

        for t in range(Tbvp):
            # We allways have the BVP signal
            parts = [x_bvp[t]]

            if t % ratio_acc == 0:
                # We have the ACC signal at this time step
                parts.append(x_acc[t // ratio_acc])

            if t % ratio_eda == 0:
                # We have the EDA signal at this time step
                parts.append(x_eda[t // ratio_eda])

            if t % ratio_hr == 0:
                # We have the HR signal at this time step
                parts.append(x_hr[t // ratio_hr])

            # Concatenate the parts to form the input for the LSTM cell
            x_in = torch.cat(parts, dim=-1)  # (batch_size, Pt) where Pt ∈ {1, ... , P}

            # Forward pass through the LSTM cell

            hx, cx = self.cells_dict[x_in.size(1)](x_in, (hx, cx))

            output.append(hx)

        output = torch.stack(output, dim=0)  # (Tbvp, batch_size, hidden_size)
        return output, (hn, cn)


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
    train_ds, valid_ds, test_ds = stratified_split(dataset, [0.8, 0.1, 0.1])
    print(f"Train: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")
    batch_size = 1024
    # train_loader = make_balanced_loader(train_ds, batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    hidden_size = 15
    num_classes = len(dataset.labels)
    print(f"Hidden size: {hidden_size}, Number of classes: {num_classes}")
    model = MultiCellLSTM(hidden_size=hidden_size, num_classes=num_classes).to(DEVICE)

    weights = compute_weights(train_ds).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weights,
        reduction="mean",
        label_smoothing=0.0,
    )
    if True:
        model_path = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=valid_loader,
            EPOCHS=250,
            model_name="multi_rate_lstm_toadstool",
            loss_fn=loss_fn,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        )
    else:
        model_path = "models/multi_rate_lstm_toadstool_20250624_111358"
    print(f"Model saved to {model_path}")
    model = MultiCellLSTM(hidden_size=hidden_size, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    # --- Evaluation on test set ---
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            # Forward pass
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
