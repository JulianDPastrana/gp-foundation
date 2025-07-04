from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(epoch_index, training_loader, model, loss_fn, optimizer):
    running_loss = 0.0

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / (i + 1)
    return avg_loss


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    EPOCHS: int,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/" + model_name + "_{}".format(timestamp))

    epoch_iter = tqdm(range(EPOCHS), desc="Training Progress")
    best_vloss = 1_000_000.0
    epoch_number = 0
    for epoch in epoch_iter:
        model.train(True)
        avg_loss = train_one_epoch(
            epoch_number, train_loader, model, loss_fn, optimizer
        )

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        epoch_iter.set_description(
            "Epoch -> Loss: {:.2e} - Val Loss: {:.2e}".format(avg_loss, avg_vloss)
        )

        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "models/" + model_name + "_{}".format(timestamp)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    return model_path
