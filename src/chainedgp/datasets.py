import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


def sine_cosine_forecasting(
    seq_length=512, num_samples=10000, Fs=128, batch_size=32, train_split=0.99
):
    """
    Generates sequences of shape (num_samples, seq_length, 2)
    and next-step targets of shape (num_samples, 2).

    Returns:
      train_loader: DataLoader yielding (X_seq, Y_next) for training
      (X_test, Y_test): tensors for evaluation
    """
    # Total timesteps needed = num_samples + seq_length
    total_steps = num_samples + seq_length
    # Time vector
    t = torch.arange(0, total_steps, dtype=torch.float32) / Fs
    # Data: [total_steps, 2]
    data = torch.stack(
        [
            torch.sin(2 * torch.pi * t) + torch.sin(6 * torch.pi * t),
            torch.cos(6 * torch.pi * t) + 0.1 * t,
        ],
        dim=-1,
    )

    # Build input sequences and next-step targets
    X = torch.stack(
        [data[i : i + seq_length] for i in range(num_samples)]
    )  # (num_samples, seq_length, 2)
    Y = data[seq_length : seq_length + num_samples]  # (num_samples, 2)

    print(
        f"Batch size: {num_samples}, Sequence length: {seq_length}, Input dim: 2, Target dim: 2"
    )

    # Split into train/test
    split_idx = int(train_split * num_samples)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]

    # Create DataLoader for training
    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return train_loader, (X_test, Y_test)


class _DualRateDataset(Dataset):
    def __init__(
        self,
        seq_length_sec: float,
        fs_high: int,
        fs_low: int,
        num_samples: int,
        device: str,
    ):
        assert fs_high % fs_low == 0, "fs_high must be a multiple of fs_low"
        self.seq_len_high = int(seq_length_sec * fs_high)
        self.ratio = fs_high // fs_low
        self.t_high = torch.arange(self.seq_len_high, dtype=torch.float32) / fs_high
        self.t_low = (
            torch.arange(self.seq_len_high // self.ratio, dtype=torch.float32) / fs_low
        )
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        phi = torch.rand(1) * 2 * torch.pi
        high = torch.sin(2 * torch.pi * self.t_high + phi)
        low = torch.cos(2 * torch.pi * self.t_low + phi)
        # Upsample by repeating values (no manual indexing)
        low_full = low.repeat_interleave(self.ratio)
        X = torch.stack([high, low_full], dim=-1).to(self.device)  # (seq_len_high, 2)
        y = (phi >= torch.pi).float().to(self.device)  # Binary target
        return X, y


def sine_cosine_dual_rate(
    seq_length_sec: float = 1.0,
    fs_high: int = 64,
    fs_low: int = 32,
    num_samples: int = 1000,
    train_split: float = 0.8,
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    Generates a dual-rate toy dataset with two-channel inputs:
      - Channel 0: high-rate sine (fs_high)
      - Channel 1: low-rate cosine (fs_low), upsampled by repetition
    Returns:
      train_loader: DataLoader yielding (X_seq, y) where X_seq is (batch_size, seq_len_high, 2)
      (X_test, y_test): tensors for evaluation
    """
    # Build dataset and split
    dataset = _DualRateDataset(seq_length_sec, fs_high, fs_low, num_samples, device)
    train_size = int(train_split * num_samples)
    test_size = num_samples - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Gather test samples into tensors
    X_test = torch.stack([test_ds[i][0] for i in range(test_size)])
    y_test = torch.tensor([test_ds[i][1] for i in range(test_size)])

    return train_loader, (X_test, y_test)
