import gpytorch as gpy
import torch


class LMCGP(gpy.models.ApproximateGP):
    def __init__(self, num_tasks, num_latents, inducing_points):
        variational_distribution = gpy.variational.NaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpy.variational.LMCVariationalStrategy(
            base_variational_strategy=gpy.variational.VariationalStrategy(
                model=self,
                inducing_points=inducing_points,
                variational_distribution=variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpy.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpy.distributions.MultivariateNormal(mean_x, covar_x)
