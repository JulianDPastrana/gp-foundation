import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchinfo import summary
from chainedgp.utils.training import train_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RegressionDataset(Dataset):
    def __init__(self, num_samples: int, device: str):
        self.num_samples = num_samples
        self.input_dim = 10
        self.output_dim = 2
        self.weigths = torch.randn(self.input_dim, self.output_dim, device=device)
        self.bias = torch.randn(self.output_dim, device=device)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim, device=DEVICE)
        noise = torch.randn(self.output_dim, device=DEVICE) * 0.1
        y = x @ self.weigths + self.bias + noise
        return x, y


def main():
    dataset = RegressionDataset(num_samples=100_000, device=DEVICE)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}"
    )

    model = torch.nn.LazyLinear(2)
    model.to(DEVICE)

    summary(model, input_size=(1, 10), verbose=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    model_name = "regression_model"
    train_model(
        model, train_loader, valid_loader, loss_fn, optimizer, model_name, EPOCHS=10
    )


if __name__ == "__main__":
    main()
