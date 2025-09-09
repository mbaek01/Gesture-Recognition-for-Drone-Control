import datetime
import os 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd 
import logging
import hashlib

from dataloader.utils import sliding_window_vote



logger = logging.getLogger(__name__)


class NormalizeSensorData:
    def __init__(self, stats):
        """
        Args:
            stats (dict): Dictionary mapping modality to (mean, std) tensors.
        """
        self.stats = stats

    def __call__(self, sample):
        for modality in ['l_cap', 'r_cap', 'l_acc', 'r_acc', 'l_gyro', 'r_gyro', 'l_quat', 'r_quat']:
            mean, std = self.stats[modality]
            sample[modality] = (sample[modality] - mean) / \
                (std + 1e-6)  # avoid division by zero
        return sample


NULL_CLASS = 'null_class'


def get_capactive_data(df, start, end):
    df = df[(df['scaled_time'] >= start) & (df['scaled_time'] <= end)
            ][['cap_0_0', 'cap_0_1', 'cap_0_2', 'cap_0_3']]
    return torch.tensor(df.values, dtype=torch.float32)


def get_acc_data(df, start, end):
    df = df[(df['scaled_time'] >= start) & (df['scaled_time'] <= end)
            ][['accelerationX', 'accelerationY', 'accelerationZ']]
    return torch.tensor(df.values, dtype=torch.float32)


def get_gyro_data(df, start, end):
    df = df[(df['scaled_time'] >= start) & (
        df['scaled_time'] <= end)][['rotationRateX', 'rotationRateY', 'rotationRateZ']]
    return torch.tensor(df.values, dtype=torch.float32)


def get_quaternion_data(df, start, end):
    df = df[(df['scaled_time'] >= start) & (df['scaled_time'] <= end)
            ][['quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']]
    return torch.tensor(df.values, dtype=torch.float32)


class SensorDataset(Dataset):

    def __init__(self, 
                 dataset_path, 
                 sessions_to_include=[], 
                 sliding_window_size=3.0, 
                 sliding_window_step=1.0, 
                 skip_null_class=True, 
                 transform=None):
        """_summary_
        TODO: skip claps label windows or not?

        """
        self.label_map = {
            'brake': 0, 'brake_fire_left': 1, 'brake_fire_right': 2, 'come_close': 3, 'cut_engine_left': 4, 'cut_engine_right': 5,
            'down': 6, 'engine_start_left': 7, 'engine_start_right': 8, 'follow': 9, 'left': 10, 'move_away': 11, 'negative': 12,
            'release_brake': 13, 'right': 14, 'slow_down': 15, 'stop': 16, 'straight': 17, 'take_photo': 18, 'up': 19, NULL_CLASS: 20, 'claps': 21,
        }
        self.sessions_to_include = sessions_to_include
        self.sliding_window_size = sliding_window_size
        self.sliding_window_step = sliding_window_step
        self.skip_null_class = skip_null_class
        
        self.transform = transform

        print("Dataset path:", dataset_path)
        self.sensor_data_path = os.path.join(
            dataset_path, 'labelstudio/final_sensor_data')
        self.label_path = os.path.join(self.sensor_data_path, 'labels')
        self.cache_path = self._get_cache_filename()

        if os.path.exists(self.cache_path):
            logger.info(
                f'Loading preprocessed dataset from cache: {self.cache_path}')
            data = torch.load(self.cache_path)
            self.l_cap_list = data['l_cap_list']
            self.r_cap_list = data['r_cap_list']
            self.l_acc_list = data['l_acc_list']
            self.r_acc_list = data['r_acc_list']
            self.l_gyro_list = data['l_gyro_list']
            self.r_gyro_list = data['r_gyro_list']
            self.l_quat_list = data['l_quat_list']
            self.r_quat_list = data['r_quat_list']
            self.label_list = data['label_list']
            return

        label_files = [x.split('.')[0][2:] for x in os.listdir(
            self.label_path) if x.startswith('S_')]
        sensor_files = os.listdir(self.sensor_data_path)
        # print(sensor_files)

        self.l_cap_list = []
        self.r_cap_list = []
        self.l_acc_list = []
        self.r_acc_list = []
        self.l_gyro_list = []
        self.r_gyro_list = []
        self.l_quat_list = []
        self.r_quat_list = []
        self.label_list = []

        print("Label files")
        print(label_files)
        print("sessions to include")
        print(sessions_to_include)

        for label_file in tqdm(label_files, desc=f"Processing label files"):
            if label_file not in self.sessions_to_include:
                logger.info(f'Skipping session {label_file}')
                continue
            label_file_path = os.path.join(
                self.label_path, f'S_{label_file}.json')
            left_watch_file = next(
                (x for x in sensor_files if x.startswith(f'{label_file}_LW')), None)
            right_watch_file = next(
                (x for x in sensor_files if x.startswith(f'{label_file}_RW')), None)
            left_glove_file = next(
                (x for x in sensor_files if x.startswith(f'{label_file}_LG')), None)
            right_glove_file = next(
                (x for x in sensor_files if x.startswith(f'{label_file}_RG')), None)
            if left_watch_file is None or right_watch_file is None or left_glove_file is None or right_glove_file is None:
                logger.error(f'Files missing for {label_file}')
                continue
            label_df = pd.read_json(label_file_path)
            # print(label_df.head())
            lw_df = pd.read_csv(os.path.join(
                self.sensor_data_path, left_watch_file))
            rw_df = pd.read_csv(os.path.join(
                self.sensor_data_path, right_watch_file))
            lg_df = pd.read_csv(os.path.join(
                self.sensor_data_path, left_glove_file))
            rg_df = pd.read_csv(os.path.join(
                self.sensor_data_path, right_glove_file))

            sliding_windows = sliding_window_vote(
                label_df, window_size=sliding_window_size, step_size=sliding_window_step, fill_label=NULL_CLASS, threshold=0.75)
            for index, row in sliding_windows.iterrows():
                start = row['window_start']
                end = row['window_end']
                label = row['label']
                if label == 'claps':
                    continue
                if label == NULL_CLASS and skip_null_class:
                    continue
                # print(f"Window: {start} - {end}, Label: {label}")

                # glove data - capacitive sensor
                l_cap_data = get_capactive_data(lg_df, start, end)
                r_cap_data = get_capactive_data(rg_df, start, end)

                # watch data - acc, gyro, quat
                l_acc_data = get_acc_data(lw_df, start, end)
                r_acc_data = get_acc_data(rw_df, start, end)
                l_gyro_data = get_gyro_data(lw_df, start, end)
                r_gyro_data = get_gyro_data(rw_df, start, end)
                l_quat_data = get_quaternion_data(lw_df, start, end)
                r_quat_data = get_quaternion_data(rw_df, start, end)

                self.l_cap_list.append(l_cap_data)
                self.r_cap_list.append(r_cap_data)
                self.l_acc_list.append(l_acc_data)
                self.r_acc_list.append(r_acc_data)
                self.l_gyro_list.append(l_gyro_data)
                self.r_gyro_list.append(r_gyro_data)
                self.l_quat_list.append(l_quat_data)
                self.r_quat_list.append(r_quat_data)
                self.label_list.append(self.label_map[label])

            # assert lengths og all lists are equal
        assert len(self.l_cap_list) == len(self.r_cap_list) == len(self.l_acc_list) == len(self.r_acc_list) == len(
            self.l_gyro_list) == len(self.r_gyro_list) == len(self.l_quat_list) == len(self.r_quat_list) == len(self.label_list)

        torch.save({
            'l_cap_list': self.l_cap_list,
            'r_cap_list': self.r_cap_list,
            'l_acc_list': self.l_acc_list,
            'r_acc_list': self.r_acc_list,
            'l_gyro_list': self.l_gyro_list,
            'r_gyro_list': self.r_gyro_list,
            'l_quat_list': self.l_quat_list,
            'r_quat_list': self.r_quat_list,
            'label_list': self.label_list,
        }, self.cache_path)
        logger.info(f'Saved preprocessed dataset to cache: {self.cache_path}')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        l_cap = self.l_cap_list[idx]
        r_cap = self.r_cap_list[idx]
        l_acc = self.l_acc_list[idx]
        r_acc = self.r_acc_list[idx]
        l_gyro = self.l_gyro_list[idx]
        r_gyro = self.r_gyro_list[idx]
        l_quat = self.l_quat_list[idx]
        r_quat = self.r_quat_list[idx]
        label = self.label_list[idx]
        sample = {
            'l_cap': l_cap,
            'r_cap': r_cap,
            'l_acc': l_acc,
            'r_acc': r_acc,
            'l_gyro': l_gyro,
            'r_gyro': r_gyro,
            'l_quat': l_quat,
            'r_quat': r_quat,
            'label': label
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def collate_fn(batch):
        l_cap = [item['l_cap'].detach().clone().to(torch.float32) for item in batch]
        r_cap = [item['r_cap'].detach().clone().to(torch.float32) for item in batch]
        l_acc = [item['l_acc'].detach().clone().to(torch.float32) for item in batch]
        r_acc = [item['r_acc'].detach().clone().to(torch.float32) for item in batch]
        l_gyro = [item['l_gyro'].detach().clone().to(torch.float32) for item in batch]
        r_gyro = [item['r_gyro'].detach().clone().to(torch.float32) for item in batch]
        l_quat = [item['l_quat'].detach().clone().to(torch.float32) for item in batch]
        r_quat = [item['r_quat'].detach().clone().to(torch.float32) for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        left_cap_padded = pad_sequence(l_cap, batch_first=True)
        right_cap_padded = pad_sequence(r_cap, batch_first=True)
        left_acc_padded = pad_sequence(l_acc, batch_first=True)
        right_acc_padded = pad_sequence(r_acc, batch_first=True)
        left_gyro_padded = pad_sequence(l_gyro, batch_first=True)
        right_gyro_padded = pad_sequence(r_gyro, batch_first=True)
        left_quat_padded = pad_sequence(l_quat, batch_first=True)
        right_quat_padded = pad_sequence(r_quat, batch_first=True)

        lengths = torch.tensor([seq.shape[0]
                               for seq in left_cap_padded], dtype=torch.long)

        return {
            'data': {'l_cap': left_cap_padded,
                     'r_cap': right_cap_padded,
                     'l_acc': left_acc_padded,
                     'r_acc': right_acc_padded,
                     'l_gyro': left_gyro_padded,
                     'r_gyro': right_gyro_padded,
                     'l_quat': left_quat_padded,
                     'r_quat': right_quat_padded,
                     'lengths': lengths},
            'label': labels
        }

    def _get_cache_filename(self):
        settings_str = str({
            'sessions': self.sessions_to_include,
            'window_size': self.sliding_window_size,
            'window_step': self.sliding_window_step,
            'skip_null_class': self.skip_null_class
        })
        hash_key = hashlib.md5(settings_str.encode()).hexdigest()

        return os.path.join(self.sensor_data_path, f'preprocessed_dataset_{hash_key}.pt')

    def compute_mean_std(self):
        """
        Compute mean and std across all sensor modalities.
        Returns:
            dict: modality -> (mean, std) tensors
        """
        modalities = ['l_cap', 'r_cap', 'l_acc', 'r_acc',
                      'l_gyro', 'r_gyro', 'l_quat', 'r_quat']
        stats = {}

        for modality in modalities:
            data = torch.cat(getattr(self, f'{modality}_list'), dim=0)
            mean = data.mean(dim=0)
            std = data.std(dim=0)
            stats[modality] = (mean, std)

        return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset = SensorDataset(
        dataset_path='/workspace/drone_gesture/full_dataset',
        sessions_to_include=['P1_S1']
    )
    stats = dataset.compute_mean_std()
    print("Mean and std for each modality:")
    for modality, (mean, std) in stats.items():
        print(f"{modality}: mean={mean.tolist()}, std={std.tolist()}")
    normalize_transform = NormalizeSensorData(stats)
    dataset = SensorDataset(
        dataset_path='/workspace/drone_gesture/full_dataset',
        sessions_to_include=['P1_S1'],
        transform=normalize_transform,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            collate_fn=SensorDataset.collate_fn)
    print(len(dataset))
    for i, data in enumerate(dataloader):
        print(data)
        break
