from torch.utils.data import DataLoader, TensorDataset
import torch


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


def sine_cosine_dual_rate(
    seq_length_sec: float = 1.0,
    fs_high: int = 64,
    fs_low: int = 32,
    num_samples: int = 1000,
    train_split: float = 0.8,
    batch_size: int = 32,
):
    """
    Generates a dual-rate toy dataset with two-channel inputs:
      - Channel 0: high-rate sine (fs_high)
      - Channel 1: low-rate cosine (fs_low), upsampled to high-rate grid with NaNs where missing
    The input X has shape (num_samples, seq_len_high, 2), and labels y are binary per sequence.

    Returns:
      train_loader: DataLoader yielding (X_seq, y) where X_seq is (batch_size, seq_len_high, 2)
      (X_test, y_test): tensors for evaluation
    """
    # Compute sequence lengths
    seq_len_high = int(seq_length_sec * fs_high)
    seq_len_low = int(seq_length_sec * fs_low)

    # Time vectors
    t_high = torch.arange(seq_len_high, dtype=torch.float32) / fs_high
    t_low = torch.arange(seq_len_low, dtype=torch.float32) / fs_low

    X_list, y_list = [], []
    for _ in range(num_samples):
        # Random phase for variation
        phi = torch.rand(1) * 2 * torch.pi
        # Channel 0: high-rate sine
        high = torch.sin(2 * torch.pi * t_high + phi)
        # Channel 1: low-rate cosine
        low = torch.cos(2 * torch.pi * t_low + phi)
        # Upsample low to high grid
        low_full = torch.full((seq_len_high,), float("nan"))
        idx_low = (t_low * fs_high).long()
        low_full[idx_low] = low
        # Stack as two channels
        X = torch.stack([high, low_full], dim=-1)  # (seq_len_high, 2)
        # Label: 1 if phase in second half, else 0
        y = (phi >= torch.pi).long()

        X_list.append(X)
        y_list.append(y)

    # Stack samples
    X = torch.stack(X_list, dim=0)  # (num_samples, seq_len_high, 2)
    y = torch.cat(y_list, dim=0)  # (num_samples,)

    # Split into train/test
    split_idx = int(train_split * num_samples)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Build DataLoader
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return train_loader, (X_test, y_test)
