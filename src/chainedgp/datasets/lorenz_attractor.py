import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class LorenzAttractorDataset(Dataset):
    def __init__(
        self,
        num_steps: int = 1000,
        initial_values: tuple[float, float, float] = (0.0, 1.0, 1.05),
        downsampling_rates: tuple[int, int, int] = (1, 10, 100),
        system_parameters: tuple[float, float, float] = (10.0, 28.0, 2.667),
        dt: float = 0.01,
        device: str = "cpu",
    ) -> None:
        # Setup
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.downsampling_rates = downsampling_rates
        s, r, b = system_parameters

        # Preallocate trajectory tensor
        traj = torch.empty((num_steps + 1, 3), device=self.device)
        traj[0] = torch.tensor(initial_values, device=self.device)

        # Unpack initial state
        x, y, z = traj[0]

        # Iterate with in-place updates to avoid extra tensor allocations
        for i in range(num_steps):
            x_dot = s * (y - x)
            y_dot = r * x - y - x * z
            z_dot = x * y - b * z

            x = x + x_dot * dt
            y = y + y_dot * dt
            z = z + z_dot * dt

            traj[i + 1, 0] = x
            traj[i + 1, 1] = y
            traj[i + 1, 2] = z

        # Downsample each channel
        self.x = traj[:-1, 0][:: downsampling_rates[0]]
        self.y = traj[:-1, 1][:: downsampling_rates[1]]
        self.z = traj[1:, 2][:: downsampling_rates[2]]

    def __len__(self) -> int:
        # Number of available samples for prediction
        return self.x.size(0)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Inputs: current x and y
        # Target: next z
        return (self.x[index], self.y[index]), self.z[index]


def main():
    dataset = LorenzAttractorDataset(num_steps=1000, downsampling_rates=(1, 2, 4))

    # Move to CPU and convert to numpy
    x = dataset.x.cpu().numpy()
    y = dataset.y.cpu().numpy()
    z = dataset.z.cpu().numpy()
    num_samples = len(dataset)
    rx, ry, rz = dataset.downsampling_rates
    print(num_samples, rx, x.shape)
    # Plot all samples in a single figure
    plt.figure()
    plt.scatter(range(0, num_samples, rx), x, label="x")
    plt.scatter(range(0, num_samples, ry), y, label="y")
    plt.scatter(range(rz, num_samples + rz, rz), z, label="z")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
