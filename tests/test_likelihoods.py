import torch
import pytest
from chainedgp.likelihoods import ChainedLikelihood
import torch.distributions as dist

@pytest.fixture
def likelihood():
    num_tasks = 3
    num_latents = 6
    return ChainedLikelihood(num_tasks=num_tasks, num_latents=num_latents)

def test_chained_likelihood_shapes(likelihood):
    batch_size = 5
    # Generate synthetic data with proper dimensions
    samples = torch.randn(batch_size, likelihood.num_tasks * 2)
    
    # Forward pass
    distribution = likelihood(samples)

    assert isinstance(distribution, dist.Independent), \
        "Output distribution must be torch.distributions.Independent"

    expected_shape = (batch_size, likelihood.num_tasks)
    assert distribution.mean.shape == expected_shape, \
        f"Means shape mismatch, expected {expected_shape}, got {distribution.mean.shape}"
    assert distribution.variance.shape == expected_shape, \
        f"Variance shape mismatch, expected {expected_shape}, got {distribution.variance.shape}"

def test_chained_likelihood_std_positive(likelihood):
    samples = torch.randn(10, likelihood.num_tasks * 2)
    distribution = likelihood(samples)

    # Check positivity of std deviations
    assert torch.all(distribution.base_dist.scale > 0), \
        "All standard deviations must be strictly positive."

