import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -*-
# Exploratory analysis script for Toadstool 2.0 multimodal dataset
# Each file: participant_{i}_toadstool_multimodal_unprocessed.npy, i = 0..9
# Sample format: ((face_seq, physio_signals, seq_start), label)
# physio_signals = [bvp (64Hz), acc (32Hz x 3 axes), eda (4Hz), hr (1Hz)]
# Labels: 0:Anger,1:Disgust,2:Fear,3:Happy,4:Sad,5:Surprised,6:Neutral

LABELS = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprised",
    6: "Neutral",
}


def load_all_participants_unprocessed(root_dir):
    """
    Load and return all participant data as a list of dicts.
    """
    data = []
    for p in range(10):
        fname = os.path.join(
            root_dir,
            f"participant_{p}/participant_{p}_toadstool_multimodal_unprocessed.npy",
        )
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Could not find file: {fname}")
        arr = np.load(fname, allow_pickle=True)
        for sample in arr:
            (face_seq, physio, seq_start), label = sample
            data.append(
                {
                    "participant": p,
                    "face_seq": face_seq,  # list of 12 frames (arrays)
                    "bvp": physio[0],  # 1D array, 64Hz * 4s = 256 samples
                    "acc": physio[1],  # 2D array (3 axes x 128 samples)
                    "eda": physio[2],  # 1D array, 4Hz * 4s = 16 samples
                    "hr": physio[3],  # 1D array, 1Hz * 4s = 4 samples
                    "seq_start": seq_start,  # integer, seconds
                    "label": label,  # integer 0-6
                }
            )
    return data


def to_dataframe(data):
    """
    Convert loaded data into a pandas DataFrame for summary statistics.
    """
    records = [
        {
            "participant": d["participant"],
            "seq_start": d["seq_start"],
            "label": d["label"],
        }
        for d in data
    ]
    return pd.DataFrame(records)


def plot_label_distribution(df):
    """
    Bar chart of overall emotion label counts.
    """
    counts = df["label"].map(LABELS).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Overall Emotion Label Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_per_participant_counts(df):
    """
    Bar chart of number of sequences per participant.
    """
    counts = df.groupby("participant").size()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Sequences per Participant")
    plt.xlabel("Participant")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_physio_signals_example(data, participant=0, index=0):
    """
    Plot example time-series of physiological signals for one sample.
    """
    sample = [d for d in data if d["participant"] == participant][index]
    time_bvp = np.linspace(0, 4, len(sample["bvp"]))
    time_acc = np.linspace(0, 4, len(sample["acc"]))
    time_eda = np.linspace(0, 4, len(sample["eda"]))
    time_hr = np.linspace(0, 4, len(sample["hr"]))

    plt.figure(figsize=(10, 8))

    # BVP
    plt.subplot(4, 1, 1)
    plt.plot(time_bvp, sample["bvp"])
    plt.title("BVP (64Hz)")
    plt.ylabel("Amplitude")

    # Accelerometer
    plt.subplot(4, 1, 2)
    plt.plot(time_acc, sample["acc"])
    plt.title("Accelerometer (32Hz)")
    plt.ylabel("g")

    # EDA
    plt.subplot(4, 1, 3)
    plt.plot(time_eda, sample["eda"])
    plt.title("EDA (4Hz)")
    plt.ylabel("µS")

    # HR
    plt.subplot(4, 1, 4)
    plt.step(time_hr, sample["hr"], where="post")
    plt.title("Heart Rate (1Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Edit this path to where your dataset is stored
    root_dir = "~/Documents/data/toadstool-dataset/toadstool2/Toadstool 2.0"
    root_dir = os.path.expanduser(root_dir)

    # Load and explore
    print("Loading data...")
    data = load_all_participants_unprocessed(root_dir)
    print(f"Total samples loaded: {len(data)}")

    df = to_dataframe(data)
    print("DataFrame head:")
    print(df.head())

    # Plot distributions
    plot_label_distribution(df)
    plot_per_participant_counts(df)

    # Plot example physiological signals for Participant 0, sample 0
    plot_physio_signals_example(data, participant=0, index=0)
