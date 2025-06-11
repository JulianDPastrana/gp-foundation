import os
import cv2
import numpy as np
import pandas as pd

LABELS = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprised",
    6: "Neutral",
}


def load_participant(root_dir, participant_id):
    """
    Load and return one participant's data as a list of dicts.
    Converts face_seq and physio signals to numpy arrays.
    Parses accelerometer strings to 2D arrays.
    """
    data = []
    fname = os.path.join(
        root_dir,
        f"participant_{participant_id}/participant_{participant_id}_toadstool_multimodal_unprocessed.npy",
    )
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Could not find file: {fname}")
    arr = np.load(fname, allow_pickle=True)
    for sample_idx, sample in enumerate(arr):
        (face_seq, physio, seq_start), label = sample

        # Convert to numpy arrays
        face_seq = np.array(face_seq)
        acc = np.array([list(map(int, s.split(";"))) for s in physio[3]])
        physio = [np.array(x) for x in physio[:3]]

        assert face_seq.shape == (12, 224, 224, 3), (
            f"Face sequence should have shape (12, 224, 224, 3), got {face_seq.shape}"
        )
        assert physio[0].shape == (256,), (
            f"BVP should have shape (256,), got {physio[0].shape}"
        )
        assert physio[1].shape == (16,), (
            f"EDA should have shape (16,), got {physio[1].shape}"
        )
        assert physio[2].shape == (4,), (
            f"HR should have shape (4,), got {physio[2].shape}"
        )
        assert acc.shape == (128, 3), (
            f"Accelerometer should have shape (128, 3), got {acc.shape}"
        )
        data.append(
            {
                "face_seq": face_seq,  # 12 frames of 224x224 RGB images
                "bvp": physio[0],  # BVP: 256 samples
                "eda": physio[1],  # EDA: 16 samples
                "hr": physio[2],  # HR: 4 samples
                "acc": acc,  # ACC: 128x3
                "seq_start": seq_start,  # Start time
                "label": label,  # Emotion label
            }
        )
    sorted_data = sorted(data, key=lambda x: x["seq_start"])
    return sorted_data


def face_seq_to_video(data, output_path, fps=3):
    """
    Reconstructs video from face sequences, overlaying only the emotion label.
    """
    face_seq_list = []
    label_list = []
    for sample in data:
        face_seq_list.append(sample["face_seq"])  # (12, H, W, C)
        label_list.append(sample["label"])  # int

    face_seq_array = np.stack(face_seq_list)  # (num_samples, 12, H, W, C)
    num_samples, num_frames, H, W, C = face_seq_array.shape
    video_frames = face_seq_array.reshape(num_samples * num_frames, H, W, C)
    labels_per_frame = np.repeat(label_list, num_frames)

    if video_frames.dtype != np.uint8:
        video_frames = np.clip(video_frames, 0, 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i, frame in enumerate(video_frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        label_text = LABELS[labels_per_frame[i]]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)
        font_thickness = 2
        outline_color = (0, 0, 0)
        x, y = 10, 40

        # Outline for better readability
        cv2.putText(
            frame_bgr,
            label_text,
            (x, y),
            font,
            font_scale,
            outline_color,
            font_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            label_text,
            (x, y),
            font,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )

        out.write(frame_bgr)
    out.release()
    print(f"Saved face sequence video to {output_path}")


def save_sequential_data(data, output_path, verbose=True):
    """
    Save the sequential data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.pop("seq_start")  # Remove seq_start for simplicity
    df.pop("face_seq")  # Remove face_seq as it's not needed in CSV
    if verbose:
        print(df.info())
    df.to_csv(output_path, index=False)
    print(f"Saved sequential data to {output_path}")


if __name__ == "__main__":
    root_dir = "~/Documents/data/toadstool-dataset/toadstool2/Toadstool 2.0"
    root_dir = os.path.expanduser(root_dir)
    cum_sum = 0
    for participant_id in range(10):
        print(f"Loading data for participant {participant_id}...")
        data = load_participant(root_dir, participant_id)
        output_path = os.path.join(
            root_dir, f"sequential_data/participant_{participant_id}.csv"
        )
        save_sequential_data(data, output_path)
        cum_sum += len(data)
        print(f"Total samples loaded: {len(data)}")

    print(f"Cumulative samples loaded: {cum_sum}")
