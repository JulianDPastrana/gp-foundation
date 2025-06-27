import os
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import WeightedRandomSampler


def load_participant(root_dir: str, participant_id: int):
    data = []
    root_dir = os.path.expanduser(root_dir)
    fname = os.path.join(
        root_dir,
        f"participant_{participant_id}",
        f"participant_{participant_id}_toadstool_multimodal_unprocessed.npy",
    )
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Could not find file: {fname}")

    arr = np.load(fname, allow_pickle=True)
    for (face_seq, physio, seq_start), label in arr:
        # Skip face_seq; parse physio & acc only
        bvp = np.array(physio[0])
        eda = np.array(physio[1])
        hr = np.array(physio[2])
        acc = np.array([list(map(int, s.split(";"))) for s in physio[3]])  # (128,3)
        data.append(
            {
                "bvp": bvp,
                "eda": eda,
                "hr": hr,
                "acc": acc,
                "seq_start": seq_start,
                "label": label,
            }
        )
    # keep chronological order
    return sorted(data, key=lambda x: x["seq_start"])


class ToadstoolSequentialDataset(Dataset):
    def __init__(self, root_dir: str, device: str):
        self.samples = []
        self.device = device
        self.labels = {
            0: "Anger",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprised",
            6: "Neutral",
        }
        for pid in range(10):  # participants 0-9
            self.samples.extend(load_participant(root_dir, pid))

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        counts = Counter(sample["label"] for sample in self.samples)
        label_dist = {self.labels[i]: counts.get(i, 0) for i in self.labels}

        # describe each modality with sample count & sampling freq over 4s
        modal_str = "\n".join(
            [
                "    • bvp : 256 samples x 1 axis →  64 Hz x 4 s per axis",
                "    • eda :  16 samples x 1 axis →   4 Hz x 4 s per axis",
                "    • hr  :   4 samples x 1 axis →   1 Hz x 4 s per axis",
                "    • acc : 128 samples × 3 axes →  32 Hz x 4 s per axis",
            ]
        )

        label_str = "\n".join(
            f"    • {name:10s}: {count:5} →  {count / self.__len__():.2%}"
            for name, count in label_dist.items()
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  num_samples: {len(self)}\n"
            f"  modalities:\n{modal_str}\n"
            f"  label_distribution:\n{label_str}\n"
            f")"
        )

    def __getitem__(self, idx):
        samp = self.samples[idx]
        x = {
            "bvp": torch.tensor(samp["bvp"], device=self.device).unsqueeze(
                -1
            ),  # shape (256,1)
            "eda": torch.tensor(samp["eda"], device=self.device).unsqueeze(
                -1
            ),  # shape (16,1)
            "hr": torch.tensor(samp["hr"], device=self.device).unsqueeze(
                -1
            ),  # shape (4,1)
            "acc": torch.tensor(samp["acc"], device=self.device),  # shape (128,3)
        }
        y = torch.tensor(samp["label"], device=self.device)  # shape (1,)
        return x, y


def stratified_split(dataset, splits=(0.8, 0.1, 0.1), seed=None):
    """
    Stratified split into train/val/test maintaining class proportions.
    """
    # Extract all labels
    labels_all = [int(dataset[i][1].item()) for i in range(len(dataset))]
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=splits[1] + splits[2], random_state=seed
    )
    train_idx, temp_idx = next(sss1.split(torch.zeros(len(labels_all)), labels_all))
    # Split temp into val/test
    labels_temp = [labels_all[i] for i in temp_idx]
    val_frac = splits[1] / (splits[1] + splits[2])
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_frac), random_state=seed
    )
    val_rel, test_rel = next(sss2.split(torch.zeros(len(labels_temp)), labels_temp))
    valid_idx = [temp_idx[i] for i in val_rel]
    test_idx = [temp_idx[i] for i in test_rel]
    return (
        Subset(dataset, train_idx),
        Subset(dataset, valid_idx),
        Subset(dataset, test_idx),
    )


def make_balanced_loader(subset, batch_size=64):
    """
    Create a DataLoader with weighted sampling to approximate equal class frequency per batch.
    """
    # Gather labels from subset
    labels = [int(subset[i][1].item()) for i in range(len(subset))]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(subset, batch_size=batch_size, sampler=sampler)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    # Example usage:
    root = "~/Documents/data/toadstool-dataset/toadstool2/Toadstool 2.0"
    dataset = ToadstoolSequentialDataset(root, device=DEVICE)
    print(dataset)
    bathch_size = 64
    loader = DataLoader(dataset, batch_size=bathch_size, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(
        batch_x["bvp"].shape,
        batch_x["eda"].shape,
        batch_x["hr"].shape,
        batch_x["acc"].shape,
        batch_y.shape,
    )
