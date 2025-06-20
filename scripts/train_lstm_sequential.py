import torch
from torch.utils.data import DataLoader, random_split

# from chainedgp.utils.training import train_model
from chainedgp.datasets.toadstool import ToadstoolSequentialDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultiCellLSTM(torch.nn.Mudule):
    def __init__(self, hidden_size: int = 10, num_classes: int = 10):
        super().__init__()
        self.hidden_size = hidden_size

        # Lets's list all possible scenarios
        # 1. BVP (64 Hz) 1 feature
        # 2. BVP + ACC (32 Hz) 4 features
        # 3. BVP + ACC + EDA (4 Hz) 5 features
        # 4. BVP + ACC + EDA + HR (1 Hz) 6 features

        # One LSTM cell for each scenario
        self.cell1 = torch.nn.LSTMCell(input_size=1, hidden_size=hidden_size)
        self.cell2 = torch.nn.LSTMCell(input_size=4, hidden_size=hidden_size)
        self.cell3 = torch.nn.LSTMCell(input_size=5, hidden_size=hidden_size)
        self.cell4 = torch.nn.LSTMCell(input_size=6, hidden_size=hidden_size)

        # Now map the hidden state to the mulit‐class output
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_classes),
            torch.nn.Softmax(dim=-1),
        )

        self.cells_dict = {
            1: self.cell1,
            4: self.cell2,
            5: self.cell3,
            6: self.cell4,
        }

        def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
            """
            x["bvp"]: (batch_size, Tbvp (256), 1)
            x["acc"]: (batch_size, Tacc (128), 3)
            x["eda"]: (batch_size, Teda (16), 1)
            x["hr"]: (batch_size, Thr (4), 1)
            """
            x_bvp = x["bvp"]  # (batch_size, Tbvp, 1)
            x_acc = x["acc"]  # (batch_size, Tacc, 3)
            x_eda = x["eda"]  # (batch_size, Teda, 1)
            x_hr = x["hr"]  # (batch_size, Thr, 1)

            batch_size, Tbvp, _ = x_bvp.shape
            _, Tacc, _ = x_acc.shape
            _, Teda, _ = x_eda.shape
            _, Thr, _ = x_hr.shape

            # Compute the up‐sampling ratios to aling wiht the BVP signal
            ratio_acc = Tbvp // Tacc  # 256 // 128 == 2
            ratio_eda = Tbvp // Teda  # 256 // 16 == 16
            ratio_hr = Tbvp // Thr  # 256 // 4 == 64

            # Initialize the chared hidden state and cell state
            hx = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
            cx = torch.zeros(batch_size, self.hidden_size, device=DEVICE)

            # Let's loop over sequences

            for t in range(Tbvp):
                # We allways have the BVP signal
                parts = [x_bvp[:, t, :]]

                if t % ratio_acc == 0:
                    # We have the ACC signal at this time step
                    parts.append(x_acc[:, t // ratio_acc, :])

                if t % ratio_eda == 0:
                    # We have the EDA signal at this time step
                    parts.append(x_eda[:, t // ratio_eda, :])

                if t % ratio_hr == 0:
                    # We have the HR signal at this time step
                    parts.append(x_hr[:, t // ratio_hr, :])

                # Concatenate the parts to form the input for the LSTM cell
                x_in = torch.cat(parts, dim=1)  # (batch_size, N) where N ∈ {1, 4, 5, 6}

                # Forward pass through the LSTM cell

                hx, cx = self.cells_dict[x_in.size(1)](x_in, (hx, cx))

            # Finally, we map the hidden state to the output
            out = self.out(hx)  # (batch_size, num_classes)
            return out


def main():
    print(f"Using device: {DEVICE}")
    root = "~/Documents/data/toadstool-dataset/toadstool2/Toadstool 2.0"
    dataset = ToadstoolSequentialDataset(root, device=DEVICE)
    print(dataset)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}"
    )
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    main()
