import pandas as pd 
import numpy as np
from collections import Counter, defaultdict


def generate_all_sessions(num_participants=11, max_sessions=5):
    return [f'P{p}_S{s}' for p in range(1, num_participants + 1) for s in range(1, max_sessions + 1)]


def generate_loso_lopo_sets(all_sessions):
    ## participants with bad results: p3, p6, p9 ,p10
    sessions_to_skip = [
        'P3_S1', 'P3_S2', 'P3_S3', 'P3_S4', 'P3_S5',
        'P4_S1', 'P4_S2', 'P4_S3', 'P4_S4', 'P4_S5',
        'P5_S4', 
        'P6_S1', 'P6_S2', 'P6_S3', 'P6_S4', 'P6_S5',
        'P7_S1', 'P7_S3',
        'P8_S4', 'P8_S5', 
        'P9_S1', 'P9_S2', 'P9_S3', 'P9_S4', 'P9_S5',
        'P10_S1', 'P10_S2', 'P10_S3', 'P10_S4', 'P10_S5',
        'P11_S2'
    ]
    # Filter sessions to remove skipped ones (now already in the correct format)
    filtered_sessions = [s for s in all_sessions if s not in sessions_to_skip]

    # LOSO
    loso_sets = []
    sessions = {s.split('_')[1] for s in filtered_sessions}
    for session in sessions:
        test_set = [s for s in filtered_sessions if s.split('_')[1] == session]
        train_set = [s for s in filtered_sessions if s.split('_')[
            1] != session]
        if test_set and train_set:
            loso_sets.append({'train': train_set, 'test': test_set})

    # LOPO
    lopo_sets = []
    participants = {s.split('_')[0] for s in filtered_sessions}
    for participant in participants:
        test_set = [s for s in filtered_sessions if s.split('_')[
            0] == participant]
        train_set = [s for s in filtered_sessions if s.split('_')[
            0] != participant]
        if test_set and train_set:
            lopo_sets.append({'train': train_set, 'test': test_set})

    return loso_sets, lopo_sets


class NormalizeSensorData:
    def __init__(self, stats):
        """
        Args:
            stats (dict): Dictionary mapping modality to (mean, std) tensors.
        """
        self.stats = stats
        self.modalities = self.stats.keys()

    def __call__(self, sample):
        for modality in self.modalities:
            mean, std = self.stats[modality]
            sample[modality] = (sample[modality] - mean) / \
                (std + 1e-6)  # avoid division by zero
        return sample


def sliding_window_vote(df, window_size=3.0, step_size=1.0, fill_label='none', threshold=None):
    """
    Applies sliding window voting on labeled intervals using majority or threshold voting.

    Args:
        df (pd.DataFrame): DataFrame with 'start', 'end', 'label' columns.
        window_size (float): Duration of each sliding window.
        step_size (float): Step between each window start.
        fill_label (str): Label to use when no match is found.
        threshold (float or None):
            - If None: majority voting.
            - If float [0,1]: use threshold-based voting by overlap duration.

    Returns:
        pd.DataFrame: DataFrame with window_start, window_end, and selected label.
    """
    start_time = df['start'].min()
    end_time = df['end'].max()
    windows = np.arange(start_time, end_time -
                        window_size + step_size, step_size)

    results = []

    for w_start in windows:
        w_end = w_start + window_size

        if threshold is None:
            # Majority voting mode
            overlaps = df[(df['start'] < w_end) & (
                df['end'] > w_start)]['label']
            if overlaps.empty:
                label = fill_label
            else:
                label = Counter(overlaps).most_common(1)[0][0]

        else:
            # Threshold-based duration overlap mode
            label_durations = defaultdict(float)

            for _, row in df.iterrows():
                label_start, label_end, label = row['start'], row['end'], row['label']
                overlap_start = max(w_start, label_start)
                overlap_end = min(w_end, label_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > 0:
                    label_durations[label] += overlap

            if label_durations:
                best_label, best_duration = max(
                    label_durations.items(), key=lambda x: x[1])
                if best_duration / window_size >= threshold:
                    label = best_label
                else:
                    label = fill_label
            else:
                label = fill_label

        results.append({
            'window_start': w_start,
            'window_end': w_end,
            'label': label
        })

    return pd.DataFrame(results)