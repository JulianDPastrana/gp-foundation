import gpytorch as gpy
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import TensorDataset, DataLoader
from gpdoc.utils.plotting import plot_multi_task_predictions

# DATASET

train_x = torch.linspace(0, 1, 100)[:, None]

train_y = torch.hstack([
    torch.sin(train_x * (2 * np.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * np.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.sin(train_x * (2 * np.pi)) + 2 * torch.cos(train_x * (2 * np.pi)) + torch.randn(train_x.size()) * 0.2,
    -torch.cos(train_x * (2 * np.pi)) + torch.randn(train_x.size()) * 0.2,
])

print(train_x.shape, train_y.shape)

input_dim = train_x.shape[-1]
num_tasks = train_y.shape[-1]

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(
        train_dataset,
        batch_size = 1024,
        shuffle = True
        )

print(f"Input dim: {input_dim}, Output dim: {num_tasks}")
# THE MODEL

num_inducing_points = 10
inducing_points = torch.rand(4, num_inducing_points, input_dim)

class LatentGPModel(gpy.models.ApproximateGP):

    def __init__(self, num_latents, num_tasks, inducing_points):

        variational_distribution = gpy.variational.NaturalVariationalDistribution(
                num_inducing_points = inducing_points.size(-2),
                batch_shape = torch.Size([num_latents])
                )

        variational_strategy = gpy.variational.LMCVariationalStrategy(
                base_variational_strategy = gpy.variational.VariationalStrategy(
                    model = self,
                    inducing_points = inducing_points,
                    variational_distribution = variational_distribution,
                    learn_inducing_locations = True
                    ),
                num_tasks = num_tasks,
                num_latents = num_latents,
                latent_dim = -1
                )

        super().__init__(variational_strategy)

        self.mean_module = gpy.means.ConstantMean(
                batch_shape = torch.Size([num_latents])
                )
        self.covar_module = gpy.kernels.ScaleKernel(
                gpy.kernels.RBFKernel(
                    batch_shape = torch.Size([num_latents])
                    ),
                batch_shape = torch.Size([num_latents])
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


## VARIATONAL APPROXIMATION

### VARIATIONAL DISTRIBUTION

### VARIATIONAL STRATEGY

### APROXIMATE MARGINAL LOG-LIKELIHOOD

## INDEPENDENT PROCESS

## MIXING STRAGEGY

model = LatentGPModel(num_latents=4,
                      num_tasks=num_tasks,
                      inducing_points=inducing_points
                      )

## LIKELIHOODS

likelihood = gpy.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

# LOSS FUNCTION

mll = gpy.mlls.VariationalELBO(
        likelihood,
        model,
        num_data = train_y.size(0)
        )

# OPTIMIZER
variational_ngd_optimizer = gpy.optim.NGD(
        model.variational_parameters(),
        num_data = train_y.size(0),
        lr = 0.1
        )

non_variational_optimizer = torch.optim.Adam(
        [
            {"params": model.hyperparameters()},
            {"params": likelihood.parameters()}
            ],
        lr = 0.01
        )
# Train the model

model.train()
likelihood.train()

num_epochs = 500
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

for i in epochs_iter:
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)

    for x_batch, y_batch in minibatch_iter:

        variational_ngd_optimizer.zero_grad()
        non_variational_optimizer.zero_grad()

        output = model(x_batch)
        loss = -mll(output, y_batch)
        
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        
        variational_ngd_optimizer.step()
        non_variational_optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpy.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

plot_multi_task_predictions(train_x, train_y, test_x, mean, lower, upper)
plt.show()
