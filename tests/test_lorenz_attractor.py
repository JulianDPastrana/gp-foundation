import torch
import pytest
from chainedgp.datasets import lorenz_attractor
from torch.utils.data import DataLoader

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BATCH_SIZE = 2
DOWSAMPLING_RATES = (1, 2, 4)


# ─── DEVICE FIXTURE ────────────────────────────────────────────────────────────
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return request.param


# ─── HELPERS ───────────────────────────────────────────────────────────────────


# ─── TESTS ─────────────────────────────────────────────────────────────────────
