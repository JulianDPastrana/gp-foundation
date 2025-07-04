import torch

loss_fn = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([20, 20, 20], dtype=torch.float32),
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
)

input_tensor = torch.tensor(
    [
        [0.0, 1e3, -1e3],  # First sample
        [1e8, 0.0, 0.0],  # Second sample
    ]
)

target_tensor = torch.tensor([2, 0])


def custom_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
):
    log_probs = torch.nn.functional.log_softmax(input, dim=1)
    loss = -log_probs[torch.arange(input.size(0)), target].mean()
    return loss


def main():
    output = loss_fn(input_tensor, target_tensor)
    print("Loss:", output.item())  # Should print the computed loss value

    output_custom = custom_cross_entropy(input_tensor, target_tensor)
    print(
        "Custom Loss:", output_custom.item()
    )  # Should print the custom computed loss value


if __name__ == "__main__":
    main()
