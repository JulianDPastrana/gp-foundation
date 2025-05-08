import gpytorch as gpy
import pytest
import torch
import torch.distributions as dist

from chainedgp.likelihoods import ChainedLikelihood


@pytest.fixture
def likelihood():
    # num_latents must be 2 * num_tasks internally
    num_tasks = 3
    num_latents = 6
    return ChainedLikelihood(num_tasks=num_tasks, num_latents=num_latents)


def test_forward_with_tensor(likelihood):
    # Called with a raw tensor of shape [batch, num_latents]
    batch_size = 20
    f = torch.randn(batch_size, likelihood.num_latents)
    conditional = likelihood(f)

    # Should be a torch.distributions.Independent over a Normal of shape [batch, num_tasks]
    assert isinstance(conditional, dist.Independent)
    assert conditional.batch_shape == (batch_size,)
    assert conditional.event_shape == (likelihood.num_tasks,)


def test_forward_with_gp_distribution(likelihood):
    # Called with a gpytorch MultiVariateNormal
    batch_size = 20
    # means in R^{batch×num_latents}
    mean = torch.ones(batch_size, likelihood.num_latents)
    # tiny diagonal covariance
    covar = torch.eye(likelihood.num_latents).expand(batch_size, -1, -1) * 1e-6
    f_dist = gpy.distributions.MultivariateNormal(mean, covar)

    # Use 5 samples to approximate the marginal
    num_samples = 5
    with gpy.settings.num_likelihood_samples(num_samples):
        marginal = likelihood(f_dist)

    # The marginal should still be a gpytorch MultivariateNormal
    assert isinstance(marginal, dist.Independent)

    # Its batch_shape should match the input batch
    assert marginal.batch_shape == (num_samples, batch_size)

    # Its event_shape should be num_tasks
    assert marginal.event_shape == (likelihood.num_tasks,)

    # The `.mean` must be either [num_samples, batch, tasks].
    mn = marginal.mean
    assert mn.dim() == 3
    # shape = [5, 20, 3]
    assert mn.size(1) == batch_size
    assert mn.size(2) == likelihood.num_tasks
    # and after averaging over samples, we get [batch, tasks]
    assert mn.mean(0).shape == (batch_size, likelihood.num_tasks)
