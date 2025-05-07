import torch
import gpytorch as gpy
import torch.distributions as dist

class ChainedLikelihood(gpy.likelihoods.Likelihood):
    def __init__(self, num_tasks, num_latents):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_latents = num_latents

    def forward(self, function_samples):
        assert function_samples.size(-1) == self.num_latents, \
            "Input size mismatch: expected twice the number of tasks."

        means = function_samples[..., ::2]
        stds = torch.nn.functional.softplus(function_samples[..., 1::2]) + 1e-6

        return dist.Independent(dist.Normal(means, stds), 1)

if __name__ == "__main__":
    # Called with tensor
    f = torch.randn(20, 6)
    likelihood = ChainedLikelihood(num_tasks=3, num_latents=6)
    conditional = likelihood(f)
    print(type(conditional), conditional.batch_shape, conditional.event_shape)

    # Called with distributions
    mean = torch.ones(20, 6)
    covar = torch.eye(6).expand(20, -1, -1) * 1e-6
    f = gpy.distributions.MultivariateNormal(mean, covar)
    with gpy.settings.num_likelihood_samples(15):
        marginal = likelihood(f)
    print(type(marginal), marginal.batch_shape, marginal.event_shape,
          marginal.mean.shape, marginal.variance.shape)
