import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader
import gpytorch as gpy


def plot_multi_task_predictions(
    train_x,
    train_y,
    test_x,
    mean,
    lower,
    upper,
    title_prefix="Task",
    sharex=False,
    sharey=False,
    figsize=None,
    ci_alpha=0.3,
    legend=True,
    colors=None,
    **fig_kwargs,
):
    def to_numpy(arr):
        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
        return np.array(arr)

    # Convert all inputs to numpy arrays
    train_x_np = to_numpy(train_x).ravel()
    test_x_np = to_numpy(test_x).ravel()
    train_y_np = to_numpy(train_y)
    mean_np = to_numpy(mean)
    lower_np = to_numpy(lower)
    upper_np = to_numpy(upper)

    # Determine grid size
    n_tasks = mean_np.shape[1]
    cols = int(np.ceil(np.sqrt(n_tasks)))
    rows = int(np.ceil(n_tasks / cols))

    # Default figsize
    if figsize is None:
        figsize = (4 * cols, 3 * rows)

    # Create subplots
    fig, axs = plt.subplots(
        rows, cols, figsize=figsize, sharex=sharex, sharey=sharey, **fig_kwargs
    )
    axs = np.atleast_1d(axs).flatten()

    # Plot each task
    for i in range(n_tasks):
        ax = axs[i]
        ax.plot(
            train_x_np,
            train_y_np[:, i],
            marker="*",
            linestyle="None",
            color=colors.get("train", "k") if colors else "k",
            label="Observed",
        )
        ax.plot(
            test_x_np,
            mean_np[:, i],
            color=colors.get("mean", "b") if colors else "b",
            label="Mean",
        )
        ax.fill_between(
            test_x_np,
            lower_np[:, i],
            upper_np[:, i],
            alpha=ci_alpha,
            color=colors.get("ci", None) if colors else None,
        )
        ax.set_title(f"{title_prefix} {i + 1}")
        if legend:
            ax.legend()

    # Turn off unused axes
    for j in range(n_tasks, len(axs)):
        axs[j].axis("off")

    fig.tight_layout()
    return fig, axs


def train_model(
    model,
    train_loader: DataLoader,
    mll,
    variational_optimizer,
    hyperparameter_optimizer,
    num_epochs: int = 300,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """
    Train a GPyTorch-style model with two optimizers, tracking loss over epochs.

    Returns
    -------
    history : dict
        {'epoch_loss': [float, ...]} average negative mll per epoch.
    """
    if device:
        model.to(device)
    history = {"epoch_loss": []}

    loop = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)

    model.train()
    for epoch in loop:
        running_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in train_loader:
            if device:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()

            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            variational_optimizer.step()
            hyperparameter_optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        history["epoch_loss"].append(avg_loss)

        if verbose:
            loop.set_postfix(loss=f"{avg_loss:.4f}")

    return history


def multi_restart_train(
    model_builder,
    likelihood_builder,
    mll_builder,
    train_loader: DataLoader,
    num_data: int,
    n_restarts: int = 5,
    num_epochs: int = 200,
    lr_var: float = 0.01,
    lr_hyp: float = 0.001,
    device=None,
    verbose=True,
    top_k: int = 3,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Performs multiple restarts of training, returns best run details.

    Args:
        model_builder: Callable to build fresh model instances.
        likelihood_builder: Callable to build fresh likelihood instances.
        mll_builder: Callable(model, likelihood) -> MLL instance.
        train_loader: DataLoader for training data.
        num_data: Total data points.
        n_restarts: Number of restarts.
        num_epochs: Training epochs per restart.
        lr_var: Learning rate for variational optimizer.
        lr_hyp: Learning rate for hyperparameter optimizer.
        device: torch device.
        verbose: If True, shows training progress.
        top_k: Top-k runs to report.

    Returns:
        best_history: Training history dict for best run.
        best_state: Dict with best model and likelihood state_dict.
        all_runs: List with all runs and results.
    """
    results = []

    for restart in range(1, n_restarts + 1):
        if verbose:
            print(f"\n== Restart {restart}/{n_restarts} ==")

        # Build fresh model and likelihood
        model = model_builder().to(device)
        likelihood = likelihood_builder()
        mll = mll_builder(model, likelihood)

        # Optimizers
        var_opt = gpy.optim.NGD(
            model.variational_parameters(), num_data=num_data, lr=lr_var
        )
        hyp_opt = torch.optim.Adam(model.hyperparameters(), lr=lr_hyp)

        # Train using imported train_model function
        history = train_model(
            model,
            train_loader,
            mll,
            variational_optimizer=var_opt,
            hyperparameter_optimizer=hyp_opt,
            num_epochs=num_epochs,
            device=device,
            verbose=verbose,
        )

        final_loss = history["epoch_loss"][-1]

        results.append(
            {
                "run": restart,
                "final_loss": final_loss,
                "history": history,
                "model_state": copy.deepcopy(model.state_dict()),
                "likelihood_state": copy.deepcopy(likelihood.state_dict()),
            }
        )

    # Sort results by final loss
    results.sort(key=lambda x: x["final_loss"])
    best_result = results[0]

    # Print top_k summary
    print(f"\nTop {top_k} runs:")
    print(f"{'Run':<5}{'Loss':>12}")
    print("-" * 17)
    for r in results[:top_k]:
        print(f"{r['run']:<5}{r['final_loss']:12.4f}")

    best_state = {
        "model": best_result["model_state"],
        "likelihood": best_result["likelihood_state"],
    }

    return best_result["history"], best_state, results
