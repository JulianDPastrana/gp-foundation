import torch
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm import tqdm

from chainedgp.datasets import sine_cosine_dual_rate as gen_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = None
print(f"Using device: {DEVICE}")


class LSTMModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 5,
        num_layers: int = 5,
        output_size: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.device = DEVICE
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = torch.nn.Linear(fc_in, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)
        y = self.fc(out[:, -1, :])
        return y


class MultiCellLSTM(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 10,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstmcellA = torch.nn.LSTMCell(
            input_size=2,
            hidden_size=self.hidden_size,
            bias=True,
        )
        self.lstmcellB = torch.nn.LSTMCell(
            input_size=1,
            hidden_size=self.hidden_size,
            bias=True,
        )
        self.linear = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=output_size,
            bias=True,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        hx = torch.zeros(x.shape[0], self.hidden_size, device=DEVICE)
        cx = torch.zeros(x.shape[0], self.hidden_size, device=DEVICE)
        output = []

        for i in range(x.shape[1]):
            if i % 2 != 0:
                # High-rate input (sine)
                hx, cx = self.lstmcellA(x[:, i, :], (hx, cx))
            else:
                # Low-rate input (cosine)
                hx, cx = self.lstmcellB(x[:, i, :1], (hx, cx))

            output.append(hx)  # Collect hidden states
        output = torch.stack(output, dim=1)  # (batch_size, seq_length, hidden_size)
        output = self.linear(output[:, -1, :])  # Use last hidden state for output
        output = self.sigmoid(output)  # Apply sigmoid activation
        return output  # (batch_size, output_size)


def view_dataset(train_loader, verbose: bool = True):
    num_samples = len(train_loader.dataset)
    x_batch, y_batch = next(iter(train_loader))
    print(f"Number of samples in dataset: {num_samples}")
    print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
    if verbose:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(x_batch[0, :, 0].cpu().numpy(), label="Sine")
        ax[1].plot(x_batch[0, ::2, 1].cpu().numpy(), label="Cosine")
        ax[0].set_title(f"Target: {y_batch[0].item()}")
        plt.show()


def train_with_validation(
    model,
    train_loader,
    val_loader=None,
    epochs: int = 150,
    lr: float = 1e-2,
    weight_decay: float = 1e-5,
    use_scheduler: bool = True,
    device: str = None,
):
    # Device setup
    model = model.to(device)

    # Loss & optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        if use_scheduler and val_loader
        else None
    )

    history = {"train_loss": [], "val_loss": []}
    epochs_iter = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    for epoch in epochs_iter:
        # ——— Training ———
        model.train()
        running_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()

            optimizer.step()
            running_train_loss += loss.item() * x_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # ——— Validation ———
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device).float().unsqueeze(1)
                    logits = model(x_val)
                    loss = criterion(logits, y_val)
                    running_val_loss += loss.item() * x_val.size(0)

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            history["val_loss"].append(epoch_val_loss)

            # step the scheduler on validation loss
            if scheduler:
                scheduler.step(epoch_val_loss)

        # ——— Logging per‐epoch ———
        if val_loader is not None:
            epochs_iter.set_postfix(
                train_loss=epoch_train_loss, val_loss=epoch_val_loss
            )
        else:
            epochs_iter.set_postfix(train_loss=epoch_train_loss)

    # ——— Plot loss curves ———
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    if val_loader is not None:
        plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return history


if __name__ == "__main__":
    train_loader, (X_test, Y_test) = gen_dataset(
        seq_length_sec=1.0,
        fs_high=64,
        fs_low=32,
        num_samples=5000,
        train_split=0.8,
        batch_size=512,
        device=DEVICE,
    )
    view_dataset(train_loader, False)
    rnn = MultiCellLSTM()
    # input_size = (512, 64, 2)
    # summary(rnn, input_size=input_size, device=DEVICE.type, verbose=1)

    # Train the model
    history = train_with_validation(
        rnn,
        train_loader,
        epochs=50,
        lr=1e-2,
        weight_decay=1e-5,
        device=DEVICE,
    )
    rnn.eval()
    # Confusion matrix
    Y_pred = (rnn(X_test) > 0.5).int().detach().cpu().numpy()
    Y_test = Y_test.detach().cpu().numpy()
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
