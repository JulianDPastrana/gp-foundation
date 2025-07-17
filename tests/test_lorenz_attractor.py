import pytest
import torch

from chainedgp.datasets.lorenz_attractor import LorenzAttractorDataset


# ─── DEVICE FIXTURE ────────────────────────────────────────────────────────────
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return request.param


# ─── TESTS ─────────────────────────────────────────────────────────────────────
def test_length_and_downsampling(device):
    num_steps = 2 * 3 * 4
    downsampling_rates = (2, 3, 4)
    # Initialize dataset on the specified device
    dataset = LorenzAttractorDataset(
        num_steps=num_steps, downsampling_rates=downsampling_rates, device=device
    )

    # Compute expected lengths for x, y, z
    expected_x_len = num_steps // downsampling_rates[0]
    expected_y_len = num_steps // downsampling_rates[1]
    expected_z_len = num_steps // downsampling_rates[2]

    # Verify tensor lengths
    assert dataset.x.size(0) == expected_x_len
    assert dataset.y.size(0) == expected_y_len
    assert dataset.z.size(0) == expected_z_len

    # __len__ should match number of samples (based on x)
    assert len(dataset) == expected_x_len


def test_getitem_returns_scalars(device):
    num_steps = 20
    downsampling_rates = (1, 1, 1)
    dataset = LorenzAttractorDataset(
        num_steps=num_steps, downsampling_rates=downsampling_rates, device=device
    )

    # Pick an arbitrary index
    idx = 5
    (x_val, y_val), z_val = dataset[idx]

    # Ensure outputs are torch Scalars
    assert isinstance(x_val, torch.Tensor) and x_val.ndim == 0
    assert isinstance(y_val, torch.Tensor) and y_val.ndim == 0
    assert isinstance(z_val, torch.Tensor) and z_val.ndim == 0


def test_repeatability(device):
    num_steps = 50
    initial_values = (0.0, 1.0, 1.05)
    downsampling_rates = (5, 5, 5)
    system_parameters = (10.0, 28.0, 2.667)
    dt = 0.01

    ds1 = LorenzAttractorDataset(
        num_steps=num_steps,
        initial_values=initial_values,
        downsampling_rates=downsampling_rates,
        system_parameters=system_parameters,
        dt=dt,
        device=device,
    )
    ds2 = LorenzAttractorDataset(
        num_steps=num_steps,
        initial_values=initial_values,
        downsampling_rates=downsampling_rates,
        system_parameters=system_parameters,
        dt=dt,
        device=device,
    )

    assert torch.allclose(ds1.x, ds2.x)
    assert torch.allclose(ds1.y, ds2.y)
    assert torch.allclose(ds1.z, ds2.z)
