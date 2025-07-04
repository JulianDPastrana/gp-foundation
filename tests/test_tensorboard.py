import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


def main():
    writer = SummaryWriter()
    r = 5
    for i in range(100):
        writer.add_scalars(
            "run_14h",
            {"xsinx": i * np.sin(i / r), "xcosx": i * np.cos(i / r), "tanx": np.tan(i / r)},
            i,
        )
    writer.close()


if __name__ == "__main__":
    main()
