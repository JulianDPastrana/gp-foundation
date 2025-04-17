import matplotlib.pyplot as plt
import numpy as np

def plot_multi_task_predictions(train_x, train_y, test_x, mean, lower, upper, title_prefix="Task"):
    """
    Plots predictions for multiple tasks with observed data, mean, and confidence intervals.

    Parameters:
    - train_x: Tensor of training inputs
    - train_y: Tensor of training outputs with shape [n_samples, n_tasks]
    - test_x: Tensor or array of test inputs
    - mean: Tensor of predictive means with shape [n_test_points, n_tasks]
    - lower: Tensor of lower confidence bounds
    - upper: Tensor of upper confidence bounds
    - title_prefix: Prefix for subplot titles (default: "Task")
    """
    n_tasks = mean.shape[1]
    grid_size = np.ceil(np.sqrt(n_tasks)).astype(int)

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(4 * grid_size, 3 * grid_size))
    axs = axs.flatten()

    for task in range(n_tasks):
        ax = axs[task]
        # Plot training data as black stars
        ax.plot(train_x.detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
        # Predictive mean as blue line
        ax.plot(test_x.numpy(), mean[:, task].numpy(), 'b')
        # Confidence interval
        ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(f'{title_prefix} {task + 1}')
    
    # Hide any unused subplots
    for i in range(n_tasks, len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()

