import torch
import math


class MultiRateLSTM(torch.nn.Module):
    def __init__(self, input_size_list: list[int], hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # Register LSTMCells in a ModuleDict keyed by input size
        self.cells = torch.nn.ModuleDict(
            {
                str(input_size): torch.nn.LSTMCell(
                    input_size=input_size, hidden_size=hidden_size
                )
                for input_size in input_size_list
            }
        )

    def forward(
        self, x: list[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        x: list of tensors with shape seq_len x batch_size x input_size
        Returns:
            output: Tensor of shape (max_len, batch_size, hidden_size)
            (h_n, c_n): final hidden and cell states
        """
        # Compute sequence lengths and maximum length
        # x = sorted(x, key=lambda seq: seq.size(0), reverse=True)
        seq_lens = [seq.size(0) for seq in x]
        max_len = max(seq_lens)
        batch_size = x[0].size(1)
        device = x[0].device
        dtype = x[0].dtype

        # Compute sampling ratios for each sequence
        ratios = [max_len // seq_len for seq_len in seq_lens]

        # Initialize hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        outputs = []

        # Iterate over timesteps
        for t in range(max_len):
            # Collect inputs from sequences at this timestep accord to their sampling rate
            xt_list = []
            for seq, ratio in zip(x, ratios):
                if t % ratio == 0:
                    idx = t // ratio
                    xt_list.append(seq[idx])
            # Concatenate along feature dimension
            xt = torch.cat(xt_list, dim=-1)

            # Select and apply the corresponding LSTMCell
            cell = self.cells[str(xt.size(1))]
            h_t, c_t = cell(xt, (h_t, c_t))
            outputs.append(h_t)

        # Stack outputs to shape (max_len, batch_size, hidden_size)
        output = torch.stack(outputs, dim=0)
        return output, (h_t, c_t)


def main():
    batch_size = 1
    input_list = [
        torch.randn(5, batch_size, 3),
        torch.randn(10, batch_size, 1),
        torch.randn(2, batch_size, 2),
    ]
    input_size_list = [6, 1, 4, 3]

    model = MultiRateLSTM(input_size_list=input_size_list, hidden_size=3)
    out, (hn, cn) = model(input_list)
    print(out.shape, hn.shape, cn.shape)


def check_padding():
    batch_size = 1
    input_list = [
        torch.ones(5, batch_size, 1),
        torch.ones(10, batch_size, 1),
        torch.ones(2, batch_size, 1),
    ]

    seq_lens = [seq.size(0) for seq in input_list]
    max_len = math.lcm(*seq_lens)
    print(max_len)

    # Compute sampling ratios for each sequence
    ratios = [max_len // seq_len for seq_len in seq_lens]
    print(ratios)

    input_dims = [seq.size(-1) for seq in input_list]
    max_dim = sum(input_dims)
    print(max_dim)

    input_mask = torch.zeros(max_dim, max_len)
    print(input_mask)


if __name__ == "__main__":
    check_padding()
