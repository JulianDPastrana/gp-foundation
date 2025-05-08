import logging

import gpytorch as gpy
import matplotlib.pyplot as plt
import torch
from gpytorch.mlls.variational_elbo import VariationalELBO
from torch.utils.data import DataLoader, TensorDataset

from chainedgp.latent_processes import LMCGP
from chainedgp.likelihoods import ChainedLikelihood
from chainedgp.utils import multi_restart_train, plot_multi_task_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = None


def generate_chained_gp_data(num_samples=500, device=DEVICE):
    X = torch.linspace(0, 4 * torch.pi, num_samples, device=device).unsqueeze(-1)
    Y = torch.hstack(
        [
            torch.normal(torch.sin(X), torch.exp(torch.cos(X))),
            torch.normal(torch.cos(X), torch.exp(torch.sin(X))),
            torch.normal(torch.sin(X / 2), torch.exp(torch.sin(2 * X))),
            torch.normal(
                torch.full_like(X, 3.0),
                torch.heaviside(torch.sin(2 * X), torch.tensor(0.0, device=device))
                + 1e-2,
            ),
        ]
    )
    return X, Y


def build_model_and_likelihood():
    num_tasks = 4
    num_latents = 2 * num_tasks
    num_inducing = 50
    num_independents = num_latents + 2
    base_inducing = torch.linspace(0, 4 * torch.pi, num_inducing, device=DEVICE)
    inducing = base_inducing.unsqueeze(-1).expand(num_independents, -1, 1)

    model = LMCGP(num_latents, num_independents, inducing_points=inducing)
    likelihood = ChainedLikelihood(num_tasks=num_tasks, num_latents=num_latents)
    return model, likelihood


def build_mll(model, likelihood, num_data):
    return VariationalELBO(likelihood, model, num_data=num_data)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info(f"Using device: {DEVICE}")
    if SEED is not None:
        logging.info(f"Random seed: {SEED}")

    train_x, train_y = generate_chained_gp_data()
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    num_data = train_y.size(0)

    # Multi-restart training
    best_history, best_state, _ = multi_restart_train(
        model_builder=lambda: build_model_and_likelihood()[0],
        likelihood_builder=lambda: build_model_and_likelihood()[1],
        mll_builder=lambda model, likelihood: build_mll(model, likelihood, num_data),
        train_loader=loader,
        num_data=num_data,
        n_restarts=10,
        num_epochs=200,
        lr_var=0.01,
        lr_hyp=0.001,
        device=DEVICE,
        verbose=True,
        top_k=5,
    )

    # Reload best model
    best_model, best_likelihood = build_model_and_likelihood()
    best_model.load_state_dict(best_state["model"])
    best_likelihood.load_state_dict(best_state["likelihood"])
    best_model.to(DEVICE)

    # Plot training curve
    plt.figure(figsize=(6, 4))
    plt.semilogy(best_history["epoch_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. Negative ELBO")
    plt.title("Best Training Run")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluation and predictions
    best_model.eval()
    with (
        torch.no_grad(),
        gpy.settings.fast_pred_var(),
        gpy.settings.num_likelihood_samples(100),
    ):
        test_x = torch.linspace(0, 4 * torch.pi, 100, device=DEVICE).unsqueeze(-1)
        latent_dist = best_model(test_x)
        pred_dist = best_likelihood(latent_dist)

        mean = pred_dist.mean.mean(0)
        lower = pred_dist.base_dist.icdf(torch.tensor([0.005], device=DEVICE)).mean(0)
        upper = pred_dist.base_dist.icdf(torch.tensor([0.995], device=DEVICE)).mean(0)

    plot_multi_task_predictions(
        train_x.squeeze(), train_y, test_x.squeeze(), mean, lower, upper
    )
    plt.show()


if __name__ == "__main__":
    main()
