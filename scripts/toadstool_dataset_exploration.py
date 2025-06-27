from collections import Counter

from torch.utils.data import DataLoader

from chainedgp.datasets.toadstool import (
    ToadstoolSequentialDataset,
    make_balanced_loader,
    stratified_split,
)

DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# Path to the dataset
root = "~/Documents/data/toadstool-dataset/toadstool2/Toadstool 2.0"
dataset = ToadstoolSequentialDataset(root, device=DEVICE)


def get_class_distribution(loader, labels):
    """
    Compute the count and percentage of each class in the given DataLoader.
    """
    counts = Counter()
    total = 0
    for _, targets in loader:
        for y in targets.view(-1):
            idx = int(y.item())
            counts[idx] += 1
            total += 1
    dist = {
        labels[idx]: {
            "count": counts.get(idx, 0),
            "percentage": (counts.get(idx, 0) / total * 100 if total > 0 else 0.0),
        }
        for idx in sorted(labels)
    }
    return dist


def print_class_distribution(dist, partition_name):
    print(f"\n{partition_name} split class distribution:")
    header = f"{'Class':<20}{'Count':>10}{'Percentage':>15}"
    print(header)
    print("-" * len(header))
    for cls, stats in dist.items():
        print(f"{cls:<20}{stats['count']:>10}{stats['percentage']:>14.2f}%")


def split_and_report_stratified(
    dataset, splits=(0.8, 0.1, 0.1), batch_size=64, seed=None
):
    print(dataset)
    train_ds, valid_ds, test_ds = stratified_split(dataset, splits, seed)
    print(f"Train: {len(train_ds)}, Validation: {len(valid_ds)}, Test: {len(test_ds)}")

    # Create balanced loaders
    train_loader = make_balanced_loader(train_ds, batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Compute and print distributions
    labels = dataset.labels
    train_dist = get_class_distribution(train_loader, labels)
    valid_dist = get_class_distribution(valid_loader, labels)
    test_dist = get_class_distribution(test_loader, labels)

    print_class_distribution(train_dist, "Train")
    print_class_distribution(valid_dist, "Validation")
    print_class_distribution(test_dist, "Test")

    for x_batch, y_batch in train_loader:
        # Count the amount of each class in the first batch
        batch_counts = Counter(y_batch.view(-1).tolist())
        print("\nFirst training batch class counts:")
        for cls, count in sorted(batch_counts.items()):
            print(f"Class {cls}: {count} samples")


if __name__ == "__main__":
    split_and_report_stratified(dataset, splits=(0.8, 0.1, 0.1), batch_size=700)
