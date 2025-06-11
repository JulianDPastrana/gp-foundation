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


def view_dataset(train_loader):
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(x_batch[0, :, 0].cpu().numpy(), label="Sine")
    ax[1].plot(x_batch[0, ::2, 1].cpu().numpy(), label="Cosine")
    ax[0].set_title(f"Target: {y_batch[0].item()}")
    plt.show()


if __name__ == "__main__":
    train_loader, (X_test, Y_test) = gen_dataset()
    view_dataset(train_loader)
