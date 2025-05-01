import gpytorch as gpy
import torch
import tqdm
from plotting import plot_multi_task_predictions
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as dist
from gpytorch.mlls.variational_elbo import VariationalELBO

# Customized ELBO to sum over all dimensions
class CustomVariationalELBO(VariationalELBO):
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs).sum(-1)

# Generate chained GP data for multi-task regression
def generate_chained_gp_data(N=500, seed=0, device="cpu"):
    # torch.manual_seed(seed)

    X = torch.linspace(0, 5, N, device=device).unsqueeze(-1)

    mean_funcs = [
        lambda x: torch.sin(2 * torch.pi * x),
        lambda x: torch.cos(2 * torch.pi * x),
        lambda x: torch.sin(4 * torch.pi * x),
        lambda x: torch.cos(4 * torch.pi * x),
    ]

    var_funcs = [
        lambda x: 0.2 * torch.cos(torch.pi * x),
        lambda x: 0.2 * torch.sin(torch.pi * x),
        lambda x: 0.5 * (x - 0.5),
        lambda x: -0.5 * x,
    ]

    means = torch.hstack([f(X) for f in mean_funcs])
    log_vars = torch.hstack([f(X) for f in var_funcs])
    stds = torch.exp(log_vars)  # sqrt(exp(log variance))

    Y = means + stds * torch.randn(N, 4, device=device)

    return X, Y

# Data preparation
train_x, train_y = generate_chained_gp_data()

input_dim = train_x.shape[-1]
num_tasks = train_y.shape[-1]
num_latents = 2 * num_tasks  # mean and variance for each task

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Inducing points setup
num_inducing_points = 50
inducing_points = torch.linspace(train_x.min(), train_x.max(), num_inducing_points).unsqueeze(-1)
inducing_points = inducing_points.expand(num_latents, -1, -1)

# Latent GP Model using LMC
class LatentGPModel(gpy.models.ApproximateGP):
    def __init__(self, num_latents, inducing_points):
        variational_distribution = gpy.variational.TrilNaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpy.variational.LMCVariationalStrategy(
            base_variational_strategy=gpy.variational.VariationalStrategy(
                model=self,
                inducing_points=inducing_points,
                variational_distribution=variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_latents,
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

# Corrected Gaussian Likelihood
class GaussianLikelihood(gpy.likelihoods.Likelihood):
    def forward(self, function_samples):
        means = function_samples[..., ::2]
        stds = torch.exp(function_samples[..., 1::2])
        return dist.Independent(dist.Normal(means, stds), 1)

# Model and likelihood
model = LatentGPModel(num_latents=num_latents, inducing_points=inducing_points)
likelihood = GaussianLikelihood()

# Variational ELBO
mll = CustomVariationalELBO(likelihood, model, num_data=train_y.size(0))

# Optimizers
variational_optimizer = gpy.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1)
hyperparameter_optimizer = torch.optim.Adam(model.hyperparameters(), lr=0.01)

# Training loop
model.train()
likelihood.train()

num_epochs = 300
for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
    for x_batch, y_batch in train_loader:
        variational_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()

        output = model(x_batch)
        loss = -mll(output, y_batch)

        loss.backward()
        variational_optimizer.step()
        hyperparameter_optimizer.step()

# Evaluation and Prediction
model.eval()
likelihood.eval()

with torch.no_grad(), gpy.settings.fast_pred_var(), gpy.settings.num_likelihood_samples(100):
    test_x = torch.linspace(train_x.min(), train_x.max(), 100).unsqueeze(-1)
    
    latent_pred_dist = model(test_x)  # MultivariateNormal (latent function)
    pred_dist = likelihood(latent_pred_dist)  # Monte Carlo samples: [samples x points x tasks]

    # Get mean and variance over the samples dimension (0)
    mean = pred_dist.mean.mean(0)  # (100, num_tasks)
    var = pred_dist.mean.var(0)    # (100, num_tasks)

    lower = mean - 1.96 * var.sqrt()
    upper = mean + 1.96 * var.sqrt()



    

plot_multi_task_predictions(train_x.squeeze(), train_y, test_x.squeeze(), mean, lower, upper)
plt.show()
