import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, random_split

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

        # LSTM cells for different scenarios
        self.cell1 = torch.nn.LSTMCell(input_size=1, hidden_size=hidden_size)
        self.cell2 = torch.nn.LSTMCell(input_size=4, hidden_size=hidden_size)
        self.cell3 = torch.nn.LSTMCell(input_size=5, hidden_size=hidden_size)
        self.cell4 = torch.nn.LSTMCell(input_size=6, hidden_size=hidden_size)

        # Batch normalization layers for different feature sizes
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.bn4 = torch.nn.BatchNorm1d(4)
        self.bn5 = torch.nn.BatchNorm1d(5)
        self.bn6 = torch.nn.BatchNorm1d(6)

        # Output mapping
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_classes),
            torch.nn.Softmax(dim=-1),
        )

        self.cells_dict = {
            1: (self.cell1, self.bn1),
            4: (self.cell2, self.bn4),
            5: (self.cell3, self.bn5),
            6: (self.cell4, self.bn6),
        }

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        x_bvp, x_acc, x_eda, x_hr = x["bvp"], x["acc"], x["eda"], x["hr"]
        batch_size, Tbvp, _ = x_bvp.shape
        ratio_acc, ratio_eda, ratio_hr = (
            Tbvp // x_acc.shape[1],
            Tbvp // x_eda.shape[1],
            Tbvp // x_hr.shape[1],
        )

        hx = torch.zeros(batch_size, self.hidden_size, device=x_bvp.device)
        cx = torch.zeros(batch_size, self.hidden_size, device=x_bvp.device)

        for t in range(Tbvp):
            parts = [x_bvp[:, t, :]]
            if t % ratio_acc == 0:
                parts.append(x_acc[:, t // ratio_acc, :])
            if t % ratio_eda == 0:
                parts.append(x_eda[:, t // ratio_eda, :])
            if t % ratio_hr == 0:
                parts.append(x_hr[:, t // ratio_hr, :])

            x_in = torch.cat(parts, dim=1)

            # Apply batch normalization
            cell, bn = self.cells_dict[x_in.size(1)]
            x_in = bn(x_in)

            hx, cx = cell(x_in, (hx, cx))

        return self.out(hx)


def main():
    print(f"Using device: {DEVICE}")
    root = "~/Documents/data/toadstool-dataset/toadstool2/Toadstool 2.0"
    dataset = ToadstoolSequentialDataset(root, device=DEVICE)
    print(dataset)
    train_ds, valid_ds, test_ds = stratified_split(dataset, [0.8, 0.1, 0.1])
    print(f"Train: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")
    batch_size = 70
    train_loader = make_balanced_loader(train_ds, batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    hidden_size = 14
    num_classes = len(dataset.labels)
    print(f"Hidden size: {hidden_size}, Number of classes: {num_classes}")
    model = MultiCellLSTM(hidden_size=hidden_size, num_classes=num_classes).to(DEVICE)

    if True:
        model_path = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=valid_loader,
            EPOCHS=500,
            model_name="multi_rate_lstm_toadstool",
            loss_fn=torch.nn.CrossEntropyLoss(),
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
