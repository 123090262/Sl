import os

import mne
import numpy as np
import torch
from mne.decoding import Scaler
from pyedflib import EdfReader
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset


NUM_STAGES = 5


def calculate_transition_matrix(label_sequences):
    """Calculate stage transitions from independent chronological sequences."""
    counts = np.zeros((NUM_STAGES, NUM_STAGES), dtype=np.float32)
    for labels in label_sequences:
        for current_stage, next_stage in zip(labels[:-1], labels[1:]):
            if 0 <= current_stage < NUM_STAGES and 0 <= next_stage < NUM_STAGES:
                counts[current_stage, next_stage] += 1

    probabilities = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)
    return torch.tensor(probabilities, dtype=torch.float32)


def get_transition_matrix(data_dict, sub_ids):
    """Build a transition prior using only the selected subjects."""
    label_sequences = []
    for sub_id in sub_ids:
        for _, labels in data_dict[sub_id]["segments"]:
            label_sequences.append(labels)
    return calculate_transition_matrix(label_sequences)


def compute_fcMatrix(X_scaled):
    channels = X_scaled.shape[1]
    flattened = X_scaled.transpose(1, 0, 2).reshape(channels, -1)
    fc_matrix = np.corrcoef(flattened)
    return np.abs(fc_matrix).astype(np.float32)


def _split_valid_segments(X_raw, labels, epoch_samples):
    """Split one recording at unknown labels so sequences remain chronological."""
    valid_indices = np.where(labels != -1)[0]
    if len(valid_indices) == 0:
        return []

    split_points = np.where(np.diff(valid_indices) != 1)[0] + 1
    segments = []
    for indices in np.split(valid_indices, split_points):
        X_segment = np.stack([
            X_raw[:, index * epoch_samples:(index + 1) * epoch_samples]
            for index in indices
        ]).astype(np.float32)
        segments.append((X_segment, labels[indices].copy()))
    return segments


def load_edf_subject(psg_path, hyp_path, channels):
    """Load one PSG/Hypnogram pair and preserve its chronological segments."""
    try:
        psg = EdfReader(psg_path)
        all_channels = psg.getSignalLabels()
        fs = int(psg.getSampleFrequency(0))
        indices = [all_channels.index(channel) for channel in channels]
        X_raw = np.vstack([psg.readSignal(index) for index in indices])
        psg._close()

        hypnogram = EdfReader(hyp_path)
        annotations = hypnogram.readAnnotations()
        hypnogram._close()

        epoch_samples = 30 * fs
        num_epochs = X_raw.shape[1] // epoch_samples
        labels = np.full((num_epochs,), -1, dtype=int)
        label_map = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,
            "Sleep stage R": 4,
        }

        for onset, duration, description in zip(*annotations):
            if description in label_map:
                start_epoch = int(onset // 30)
                end_epoch = min(start_epoch + int(duration // 30), num_epochs)
                labels[start_epoch:end_epoch] = label_map[description]

        sleep_indices = np.where(labels > 0)[0]
        if len(sleep_indices) > 0:
            buffer_epochs = 60
            start_index = max(0, sleep_indices[0] - buffer_epochs)
            end_index = min(num_epochs, sleep_indices[-1] + buffer_epochs + 1)
            X_raw = X_raw[:, start_index * epoch_samples:end_index * epoch_samples]
            labels = labels[start_index:end_index]

        segments = _split_valid_segments(X_raw, labels, epoch_samples)
        original_labels = labels[labels != -1]
        return segments, original_labels, fs

    except Exception as error:
        print(f"Error loading {psg_path}: {error}")
        return None, None, None


def apply_custom_filter(X, fs):
    nyquist = 0.5 * fs
    eeg_b, eeg_a = butter(4, [0.5 / nyquist, 35.0 / nyquist], btype="band")
    X[:, 0:2, :] = filtfilt(eeg_b, eeg_a, X[:, 0:2, :], axis=-1)
    eog_b, eog_a = butter(4, [0.3 / nyquist, 10.0 / nyquist], btype="band")
    X[:, 2, :] = filtfilt(eog_b, eog_a, X[:, 2, :], axis=-1)
    return X


def apply_subject_scaler_mne(X_sub, fs, channels):
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types=["eeg", "eeg", "eog"])
    scaler = Scaler(info=info, scalings="mean")
    return scaler.fit_transform(X_sub).astype(np.float32)


def get_data_dict(root, sub_list, channels):
    """Load subjects and keep each recording segment separate."""
    data_dict = {}
    all_original_labels = []
    all_files = os.listdir(root)

    for sub_id in sub_list:
        psg_files = sorted([
            filename
            for filename in all_files
            if filename.startswith(sub_id) and filename.endswith("PSG.edf")
        ])
        subject_segments = []
        subject_fs = None

        for psg_filename in psg_files:
            file_prefix = psg_filename[:6]
            hypnogram_files = [
                filename
                for filename in all_files
                if filename.startswith(file_prefix) and filename.endswith("Hypnogram.edf")
            ]
            if not hypnogram_files:
                continue

            segments, original_labels, fs = load_edf_subject(
                os.path.join(root, psg_filename),
                os.path.join(root, hypnogram_files[0]),
                channels,
            )
            if segments is not None:
                subject_segments.extend(segments)
                all_original_labels.append(original_labels)
                subject_fs = fs

        if not subject_segments:
            continue

        segment_lengths = [len(labels) for _, labels in subject_segments]
        X_combined = np.concatenate([X for X, _ in subject_segments], axis=0)
        X_filtered = apply_custom_filter(X_combined, subject_fs)
        X_scaled = apply_subject_scaler_mne(X_filtered, subject_fs, channels)
        A_fc = compute_fcMatrix(X_scaled)

        scaled_segments = []
        start = 0
        for (_, labels), length in zip(subject_segments, segment_lengths):
            scaled_segments.append((X_scaled[start:start + length], labels))
            start += length

        data_dict[sub_id] = {
            "segments": scaled_segments,
            "A_fc": A_fc,
        }
        print(f"Loaded {sub_id}, valid epochs: {len(X_scaled)}, segments: {len(scaled_segments)}")

    P_matrix = get_transition_matrix(data_dict, data_dict.keys())
    y_orig_total = (
        np.concatenate(all_original_labels, axis=0)
        if all_original_labels
        else np.array([])
    )
    return data_dict, P_matrix, y_orig_total


class SeqSleepDataset(Dataset):
    """Return centered continuous sequences without crossing segment boundaries."""

    def __init__(self, data_dict, sub_ids, seq_len):
        if seq_len <= 0 or seq_len % 2 == 0:
            raise ValueError("seq_len must be a positive odd integer")

        self.radius = seq_len // 2
        self.samples = []
        self.segment_dict = {}
        self.A_dict = {}

        for sub_id in sub_ids:
            self.segment_dict[sub_id] = data_dict[sub_id]["segments"]
            self.A_dict[sub_id] = data_dict[sub_id]["A_fc"].astype(np.float16)
            for segment_index, (_, labels) in enumerate(self.segment_dict[sub_id]):
                for center_index in range(len(labels)):
                    self.samples.append((sub_id, segment_index, center_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sub_id, segment_index, center_index = self.samples[index]
        X_segment, y_segment = self.segment_dict[sub_id][segment_index]
        indices = np.arange(
            center_index - self.radius,
            center_index + self.radius + 1,
        )
        indices = np.clip(indices, 0, len(y_segment) - 1)

        return (
            torch.tensor(X_segment[indices], dtype=torch.float32),
            torch.tensor(y_segment[indices], dtype=torch.long),
            torch.tensor(self.A_dict[sub_id], dtype=torch.float32),
        )


if __name__ == "__main__":
    root = r"E:\EEG\dataset\sleep-edf-database-expanded-1.0.0\SleepEDF-20"
    stage_names = ["W", "N1", "N2", "N3", "REM"]
    sub_list = sorted({f[:5] for f in os.listdir(root) if "PSG.edf" in f})[:20]

    data_dict, P_matrix, y_orig_total = get_data_dict(
        root,
        sub_list,
        channels=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
    )

    print("\n--- Valid chronological epoch statistics ---")
    for index, name in enumerate(stage_names):
        print(f"{name}: {int((y_orig_total == index).sum())}")
    print(f"Total Valid: {len(y_orig_total)}")

    print("\nTransition probability matrix P")
    P = P_matrix.cpu().numpy()
    print("      " + "  ".join([f"{name:>6s}" for name in stage_names]))
    for index, row_name in enumerate(stage_names):
        print(f"{row_name:>4s}  " + "  ".join([f"{P[index, j]:6.4f}" for j in range(NUM_STAGES)]))
