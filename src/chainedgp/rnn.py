import torch
import numpy as np
import matplotlib.pyplot as plt


class MultiRateLSTM(torch.nn.Module):
    def __init__(
        self, input_size_list: list[int], hidden_size: int, num_layers: int, device: str
    ) -> None:
        super().__init__()
        self.input_size_list = input_size_list
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # One LSTMCell by scenario
        self.cells_dict = {
            input_size: torch.nn.LSTMCell(
                input_size=input_size, hidden_size=hidden_size
            )
            for input_size in input_size_list
        }

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x_sorted = sorted(x, key=lambda seq: seq.size(0), reverse=True)
        seq_len_list = [seq.size(0) for seq in x_sorted]
        max_len, batch_size, _ = x_sorted[0].shape
        ratios = [seq_len_list[0] // seq_len for seq_len in seq_len_list]

        condition_indices = [
            {t for t in range(max_len) if t % ratio == 0} for ratio in ratios
        ]
        print(condition_indices)
        hx = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=self.device
        )
        cx = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=self.device
        )
        h_t_minus_1 = hx
        h_t = hx
        c_t_minus_1 = cx
        c_t = cx
        output = []

        for t in range(max_len):
            x_t = [
                seq[t // ratio]
                for seq, ratio, cond_set in zip(x_sorted, ratios, condition_indices)
                if t in cond_set
            ]

            x_t = torch.cat(x_t, dim=-1)
            cell = self.cells_dict[x_t.size(1)]
            for layer in self.num_layers:
                h_t[layer], c_t[layer]
            print(x_t)

        # for t in range(Tbvp):
        #     # We allways have the BVP signal
        #     parts = [x_bvp[t]]
        #
        #     if t % ratio_acc == 0:
        #         # We have the ACC signal at this time step
        #         parts.append(x_acc[t // ratio_acc])
        #
        #     if t % ratio_eda == 0:
        #         # We have the EDA signal at this time step
        #         parts.append(x_eda[t // ratio_eda])
        #
        #     if t % ratio_hr == 0:
        #         # We have the HR signal at this time step
        #         parts.append(x_hr[t // ratio_hr])
        #
        #     # Concatenate the parts to form the input for the LSTM cell
        #     x_in = torch.cat(parts, dim=-1)  # (batch_size, Pt) where Pt ∈ {1, ... , P}
        #
        #     # Forward pass through the LSTM cell
        #
        #     hx, cx = self.cells_dict[x_in.size(1)](x_in, (hx, cx))
        #
        #     output.append(hx)
        #
        # output = torch.stack(output, dim=0)  # (Tbvp, batch_size, hidden_size)
        # return output, (hn, cn)


def main():
    batch_size = 2

    input_list = [
        torch.randn(5, batch_size, 3),
        torch.randn(10, batch_size, 1),
        torch.randn(2, batch_size, 2),
    ]

    input_size_list = [1, 4, 6]

    rnn = MultiRateLSTM(
        input_size_list=input_size_list, hidden_size=3, num_layers=1, device="cpu"
    )

    rnn(input_list)


if __name__ == "__main__":
    main()
