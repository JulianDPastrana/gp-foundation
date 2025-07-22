import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class LorenzAttractorDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 3,
        window_steps: int = 4,
        dt: float = 0.01,
        downsampling_rates: tuple[int, int] = (1, 2),
        initial_values: tuple[float, float, float] = (0.0, 1.0, 1.05),
        system_parameters: tuple[float, float, float] = (10.0, 28.0, 2.667),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.device = torch.device(device)
        self.window_steps = window_steps
        self.dx, self.dy = downsampling_rates
        s, r, b = system_parameters

        # 1) integrate full trajectory
        num_steps = window_steps * num_samples + 1
        traj = torch.empty((num_steps, 3), device=self.device)
        traj[0] = torch.tensor(initial_values, device=self.device)
        x, y, z = traj[0]
        for i in range(num_steps - 1):
            x_dot = s * (y - x)
            y_dot = r * x - y - x * z
            z_dot = x * y - b * z
            x = x + x_dot * dt
            y = y + y_dot * dt
            z = z + z_dot * dt
            traj[i + 1] = torch.stack((x, y, z))

        self.traj = traj

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        w = self.window_steps
        # sequence of x and y, downsampled
        x_seq = self.traj[index : index + w : self.dx, 0]
        y_seq = self.traj[index : index + w : self.dy, 1]
        # target is z at future time t+w
        z_target = self.traj[index + w + 1, 2]
        return (x_seq, y_seq), z_target


def main():
    dataset = LorenzAttractorDataset(num_samples=100, window_steps=10)
    num_samples = len(dataset)
    window_steps = dataset.window_steps
    rx, ry = dataset.dx, dataset.dy
    print(f"Num of samples: {num_samples}")
    num_axis = min(3, num_samples)
    fig, _ = plt.subplots(num_axis, 1, figsize=(12, 4 * num_samples))

    for idx, ax in enumerate(fig.axes):
        (x_seq, y_seq), z_seq = dataset[idx]
        if idx == 0:
            print(f"x seq len: {len(x_seq)}, x seq len: {len(y_seq)}")

        ax.scatter(range(0, window_steps, rx), x_seq.cpu().numpy(), label="x")
        ax.scatter(range(0, window_steps, ry), y_seq.cpu().numpy(), label="y")
        ax.scatter([window_steps], z_seq.cpu().numpy(), label="z")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
