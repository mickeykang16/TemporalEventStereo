from pathlib import Path

import torch
import pdb
from .dsecSequence import Sequence

DATA_SPLIT = {
    'train': ['interlaken_00_c', 'interlaken_00_d', 'interlaken_00_e', 'zurich_city_00_a',
              'zurich_city_00_b', 'zurich_city_01_a', 'zurich_city_01_b', 'zurich_city_01_c',
              'zurich_city_01_d', 'zurich_city_01_e', 'zurich_city_02_a',
              'zurich_city_02_b', 'zurich_city_02_c', 'zurich_city_02_d', 'zurich_city_02_e',
              'zurich_city_03_a', 'zurich_city_04_a', 'zurich_city_04_b', 'zurich_city_04_c',
              'zurich_city_04_d', 'zurich_city_04_e', 'zurich_city_04_f', 'zurich_city_09_a',
              'zurich_city_09_b',  'zurich_city_09_e', 'zurich_city_10_a',
              'zurich_city_11_a',  'zurich_city_11_c',
              'interlaken_00_f', 'interlaken_00_g', 'thun_00_a', 'zurich_city_05_a',
                'zurich_city_05_b', 'zurich_city_07_a',
              'zurich_city_09_d', 'zurich_city_10_b'],
    'validation': ['zurich_city_01_f', 'zurich_city_06_a', 'zurich_city_11_b','zurich_city_09_c',
                'zurich_city_08_a']
    }

DATA_SPLIT_MINI = {
    'train': ['zurich_city_03_a', 'zurich_city_04_a', 'zurich_city_04_b', 'zurich_city_04_c',
              'zurich_city_04_d', 'zurich_city_04_e', 'zurich_city_04_f', 'zurich_city_09_a',
             ],
    'validation': ['zurich_city_01_f', 'zurich_city_06_a']
    }

DATA_SPLIT_SUPER_MINI = {
    'train': ['zurich_city_04_a' ],
    'validation': ['zurich_city_04_d']
    }

class DatasetProvider:
    def __init__(self, dataset_path: Path,  raw_dataset_path: Path, delta_t_ms: int=100, num_bins=15, frame_idxs = range(-3, 1), 
                 eval_maxdisp=192, pseudo_path=None, pad_width=648, pad_height=480, use_mini = False, use_super_mini = False, img_load = False):

        train_path = dataset_path / 'train'
        train_raw_path = raw_dataset_path / 'train'
        
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_raw_path.is_dir(), str(train_raw_path)
        assert train_path.is_dir(), str(train_path)
        
        if use_super_mini:
            data_split = DATA_SPLIT_SUPER_MINI
        elif use_mini:
            data_split = DATA_SPLIT_MINI
        else:
            data_split = DATA_SPLIT

        test_path = dataset_path / 'test'
        test_raw_path = raw_dataset_path / 'test'
        
        assert test_path.is_dir(), str(test_path)

        test_sequences = list()
        for child in test_path.iterdir():
            if str(child).split('/')[-1] not in 'thun_02_a':
                raw_child = test_raw_path / str(child).split('/')[-1]
                test_sequences.append(Sequence(child, 'test', delta_t_ms, num_bins, frame_idxs, eval_maxdisp, pseudo_path=pseudo_path,  pad_width=pad_width, pad_height=pad_height,
                                               raw_seq_path = raw_child, img_load = img_load))

        
        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset