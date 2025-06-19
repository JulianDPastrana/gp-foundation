import torch
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset, random_split
from chainedgp.utils.training import train_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = None
print(f"Using device: {DEVICE}")


class MultiRateSequency(Dataset):
    def __init__(
        self,
        seq_length_sec: float,
        freq_list: int,
        num_samples: int,
        device: str,
    ):
        time_list = []
        for freq in freq_list:
            t = torch.arange(0, seq_length_sec, 1 / freq, device=device)
            time_list.append(t)

        self.time_list = time_list
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sec = torch.randint(0, 2, (1,)).item() * 2 - 1
        x1 = torch.sin(2 * torch.pi * self.time_list[0])
        x2 = torch.sin(2 * torch.pi * self.time_list[1] - sec * 2 * torch.pi / 3)
        x3 = torch.sin(2 * torch.pi * self.time_list[2] - sec * 4 * torch.pi / 3)
        # Stack the inputs of different sizes
        x = [x1, x2, x3]
        y = torch.tensor([(sec + 1) / 2]).to(self.device)
        return x, y


class MultiCellLSTM(torch.nn.Module):
    def __init__(self, hidden_size: int = 10, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell3 = torch.nn.LSTMCell(input_size=3, hidden_size=hidden_size)
        self.cell2 = torch.nn.LSTMCell(input_size=2, hidden_size=hidden_size)
        self.cell1 = torch.nn.LSTMCell(input_size=1, hidden_size=hidden_size)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3 = x_list
        B, T1 = x1.shape
        _, T2 = x2.shape
        _, T3 = x3.shape

        # Compute the up‐sampling ratios
        ratio2 = T1 // T2  # e.g. 128//64 == 2
        ratio3 = T1 // T3  # e.g. 128//32 == 4

        # initialize hidden/cell states
        hx = torch.zeros(B, self.hidden_size, device=DEVICE)
        cx = torch.zeros(B, self.hidden_size, device=DEVICE)

        for t in range(T1):
            # always have x1 at every t
            inputs = [x1[:, t].unsqueeze(1)]  # shape (B,1)

            # if this time-step aligns with x2, pull from x2
            if t % ratio2 == 0:
                idx2 = t // ratio2
                inputs.append(x2[:, idx2].unsqueeze(1))

            # if it aligns with x3, pull from x3
            if t % ratio3 == 0:
                idx3 = t // ratio3
                inputs.append(x3[:, idx3].unsqueeze(1))

            x_in = torch.cat(inputs, dim=1)  # shape (B, N) where N∈{1,2,3}

            # route to the right LSTMCell
            if x_in.size(1) == 3:
                hx, cx = self.cell3(x_in, (hx, cx))
            elif x_in.size(1) == 2:
                hx, cx = self.cell2(x_in, (hx, cx))
            else:  # 1
                hx, cx = self.cell1(x_in, (hx, cx))

        return self.out(hx)  # shape (B, output_size)


def visualize_dataset(data_loader: DataLoader):
    index = torch.randint(0, batch_size, (1,)).item()
    x_batch, y_batch = next(iter(data_loader))
    fig = plt.figure(figsize=(10, 6))
    for i in range(3):
        fig.add_subplot(3, 1, i + 1)
        plt.plot(x_batch[i][index].cpu().numpy(), label="Phase {i + 1}")
        plt.title(f"Target: {y_batch[index].item()}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = MultiRateSequency(
        seq_length_sec=2.0, freq_list=[64, 32, 4], num_samples=100_000, device=DEVICE
    )
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}"
    )
    # visualize_dataset(train_loader)
    model = MultiCellLSTM(hidden_size=10, output_size=1).to(DEVICE)
    if True:
        model_path = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=valid_loader,
            EPOCHS=10,
            model_name="multi_rate_lstm",
            loss_fn=torch.nn.BCELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        )
    else:
        model_path = "models/multi_rate_lstm_20250619_165956"
    print(f"Model saved to {model_path}")
    model = MultiCellLSTM(hidden_size=10, output_size=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    # --- Evaluation on test set ---
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            # Move inputs & labels to device
            if isinstance(x, (list, tuple)):
                x = [xi.to(DEVICE) for xi in x]
            else:
                x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Forward pass
            out = model(x).squeeze()  # shape: (batch,)
            preds = (out > 0.5).long()  # binary predictions 0 or 1

            # Accumulate
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
