import gpytorch as gpy
import torch
import torch.distributions as dist


class ChainedLikelihood(gpy.likelihoods.Likelihood):
    def __init__(self, num_tasks, num_latents):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_latents = num_latents

    def forward(self, function_samples):
        assert function_samples.size(-1) == self.num_latents, (
            "Input size mismatch: expected twice the number of tasks."
        )

        means = function_samples[..., ::2]
        stds = torch.nn.functional.softplus(function_samples[..., 1::2]) + 1e-6

        return dist.Independent(dist.Normal(means, stds), 1)
