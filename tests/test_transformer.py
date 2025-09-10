import math
import random
from typing import Any, Dict, Callable

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


class BinaryTimeSeriesDataset(Dataset):
    """
    Generates synthetic pairs of signals over a 4-second window:
      - High-res: fs_hi = 64 Hz  -> 256 points
      - Low-res:  fs_lo = 16 Hz  ->  64 points

    The low-res series is inserted sparsely into a 256-length array at indices
    [0, 4, 8, ...] (every 4th sample), with zeros elsewhere. A mask marks these
    true low-res sample positions (1 at true samples, 0 in the gaps).
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        window_sec: float = 4.0,
        fs_hi: int = 64,
        fs_lo: int = 16,
        seed: int = 1234,
        device: str = "cpu",
    ):
        super().__init__()
        assert window_sec > 0
        assert fs_hi % fs_lo == 0, "fs_hi must be an integer multiple of fs_lo"
        self.n = n_samples
        self.T = window_sec
        self.fs_hi = fs_hi
        self.fs_lo = fs_lo
        self.r = fs_hi // fs_lo  # upsample ratio (here 4)
        self.N_hi = int(self.T * self.fs_hi)  # 256
        self.N_lo = int(self.T * self.fs_lo)  # 64
        self.t_hi = torch.arange(0, self.N_hi, device=device) / self.fs_hi
        self.t_lo = torch.arange(0, self.N_lo, device=device) / self.fs_lo
        self.device = device

        random.seed(seed)
        torch.manual_seed(seed)

        # Precompute indices where the low-res samples land in the high-res grid
        self.lo_at_hi_idx = torch.arange(
            0, self.N_hi, self.r, device=device
        )  # [0,4,8,...]

    def __len__(self) -> int:
        return self.n

    def _make_continuous_signal(
        self, label: int
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Create a continuous-time function s(t) that encodes the class.
        Class 0: lower-band content (e.g., 0.5–2.5 Hz)
        Class 1: higher-band content (e.g., 3.5–6 Hz)
        Add mild amplitude modulation + noise for variety.
        """
        # Frequencies by class
        if label == 0:
            f = torch.empty(1).uniform_(0.5, 2.5).item()
        else:
            f = torch.empty(1).uniform_(3.5, 6.0).item()

        phi = torch.empty(1).uniform_(0, 2 * math.pi).item()
        amp = torch.empty(1).uniform_(0.8, 1.2).item()

        def s(t: torch.Tensor) -> torch.Tensor:
            carrier = amp * torch.sin(2 * math.pi * f * t + phi)
            # gentle AM + a small trend to avoid trivial memorization
            am = 0.15 * torch.sin(2 * math.pi * 0.25 * t)  # slow modulation
            trend = 0.05 * (t - self.T / 2)
            return (1.0 + am) * carrier + trend

        return s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = random.randint(0, 1)
        s = self._make_continuous_signal(label)

        # Sample the same underlying function at both rates (aligned time frame)
        x_hi = s(self.t_hi) + 0.05 * torch.randn(self.N_hi, device=self.device)
        x_lo = s(self.t_lo) + 0.05 * torch.randn(self.N_lo, device=self.device)

        # Place low-res samples sparsely into a 256-length vector; zeros elsewhere
        x_lo_sparse = torch.zeros(self.N_hi, device=self.device)
        x_lo_sparse[self.lo_at_hi_idx] = x_lo

        # Create data matrix
        data = torch.stack([x_hi, x_lo_sparse], dim=0)

        mask = torch.ones_like(data, device=self.device)
        mask[1, self.lo_at_hi_idx] = 0.0

        # Return tensors ready to be batched
        return {
            "data": data.float(),  # [2, 256]
            "mask": mask.float(),  # [2, 256]
            "y": torch.tensor(label, dtype=torch.long),
        }


def collate_batch(batch):
    """
    Stacks samples into:
      inputs: [B, C=3, 256]  where channels = [x_hi, x_lo_sparse, mask_lo]
      labels: [B]
    """
    x_hi = torch.stack([b["x_hi"] for b in batch], dim=0)  # [B, 256]
    x_lo_sp = torch.stack([b["x_lo_sparse"] for b in batch], dim=0)  # [B, 256]
    mask_lo = torch.stack([b["mask_lo"] for b in batch], dim=0)  # [B, 256]
    y = torch.stack([b["y"] for b in batch], dim=0)  # [B]

    # Combine channels -> [B, 3, 256]
    inputs = torch.stack([x_hi, x_lo_sp, mask_lo], dim=1)
    return inputs, y


def make_dataloader(
    n_samples: int = 10_000,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
) -> DataLoader:
    ds = BinaryTimeSeriesDataset(n_samples=n_samples, device=device)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # collate_fn=collate_batch,
    )


def exaple_and_view() -> None:
    loader = make_dataloader(n_samples=512, batch_size=32, device="cpu")
    xb, yb = next(iter(loader))
    print("inputs:", xb.shape, "(B, C=3, T=256)")
    print("labels:", yb.shape, yb.dtype)
    # xb[:, 2, :] is the mask for the low-res channel's true sample positions

    plt.imshow(xb[0, :, :50])
    plt.show()


def test_transformer() -> None:
    transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    print(out.shape)


if __name__ == "__main__":
    # exaple_and_view()
    test_transformer()
