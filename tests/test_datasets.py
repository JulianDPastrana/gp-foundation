import torch
import pytest

from chainedgp.datasets import sine_cosine_dual_rate


@pytest.mark.parametrize(
    "seq_length_sec, fs_high, fs_low, num_samples, train_split, batch_size",
    [
        (1.0, 64, 32, 100, 0.8, 16),
        (0.5, 8, 4, 20, 0.5, 5),
    ],
)
def test_toy_dataset_dual_rate_shapes_and_nans(
    seq_length_sec, fs_high, fs_low, num_samples, train_split, batch_size
):
    # Generate dataset
    train_loader, (X_test, y_test) = sine_cosine_dual_rate(
        seq_length_sec=seq_length_sec,
        fs_high=fs_high,
        fs_low=fs_low,
        num_samples=num_samples,
        train_split=train_split,
        batch_size=batch_size,
    )

    # Compute expected sequence lengths
    seq_len_high = int(seq_length_sec * fs_high)
    seq_len_low = int(seq_length_sec * fs_low)
    expected_nans = seq_len_high - seq_len_low

    # Check test split shapes
    num_train = int(train_split * num_samples)
    num_test = num_samples - num_train
    assert X_test.shape == (num_test, seq_len_high, 2)
    assert y_test.shape == (num_test,)

    # Fetch one batch
    X_batch, y_batch = next(iter(train_loader))

    # Check batch shapes
    assert X_batch.shape == (batch_size, seq_len_high, 2)
    assert y_batch.shape == (batch_size,)

    # Check label types and values
    assert y_batch.dtype == torch.long
    assert torch.all((y_batch == 0) | (y_batch == 1))

    # Check NaNs count in low-rate channel
    nan_counts = torch.isnan(X_batch[..., 1]).sum(dim=1)
    # All samples in batch should have same number of NaNs
    assert torch.all(nan_counts == expected_nans)


if __name__ == "__main__":
    pytest.main([__file__])
