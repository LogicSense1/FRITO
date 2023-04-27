import io
import os
import pathlib
import random

import av
import librosa
import torchaudio
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DistributedSampler, WeightedRandomSampler
from nsynth.helpers.audiodatasets import FilesCachedDataset, SimpleSelectionDataset
import torch
from ba3l.ingredients.datasets import Dataset
import pandas as pd
from sacred.config import DynamicIngredient, CMD
from scipy.signal import convolve
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import h5py
from helpers.audiodatasets import  PreprocessDataset


LMODE = os.environ.get("LMODE", False)

dataset = Dataset('nsynth')


@dataset.config
def default_config():
    name = 'nsynth'  # dataset name
    # processed audio content files cached in this location
    cache_root_path = ""
    # base directory of the dataset as downloaded and reassembled
    base_dir = "./audioset_hdf5s/nsynth"
    audio_path = os.path.join(base_dir, "audio")
    audio_processor = DynamicIngredient(path="nsynth.helpers.audioprocessors.base.default_processor")
    process_func = CMD(".audio_processor.get_processor_default")
    train_files_json = os.path.join(base_dir, "train.json")
    test_files_json = os.path.join(base_dir, "valid.json")
    meta_json = os.path.join(base_dir, "all.json")
    use_full_dev_dataset = 0
    subsample = 0
    num_of_classes = 10


@dataset.config
def process_config():
    # audio processor capable of creating spectrograms -
    # in this example we only use it for resampling and
    # create the spectrograms on the fly in teachers_gang.py
    audio_processor = dict(sr=32000,
                           resample_only=True)


@dataset.command
class BasicNSYNTHDataset(TorchDataset):
    """
    Basic NSYNTH Dataset
    """

    def __init__(self, meta_json):
        """
        @param meta_json: meta json file for the dataset
        return: name of the file, label, device and cities
        """
        df = pd.read_json(meta_json)
        df = df.transpose()
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(df[['instrument_family_str']].values.reshape(-1))
        self.source = le.fit_transform(df[['instrument_source_str']].values.reshape(-1))
        self.files = df[['note_str']].values.reshape(-1)
        df.index = [i for i in range(len(self.labels))]
        self.df = df

    def __getitem__(self, index):
        return self.files[index], self.labels[index], self.source[index]

    def __len__(self):
        return len(self.files)


@dataset.command
class ContentDataset(TorchDataset):
    """
    gets the audio content from files using audioprocessor
    meta_json: meta file containing index
    audio_path: audio path to the files
    process_function: function used to process raw audio
    return: x, file name, label, device, city
    """
    def __init__(self, meta_json, audio_path, process_func):
        self.ds = BasicNSYNTHDataset(meta_json)
        self.process_func = process_func
        self.audio_path = audio_path

    def __getitem__(self, index):
        file, label, source = self.ds[index]
        file += '.wav'
        x = self.process_func(os.path.join(self.audio_path, file))
        return x, file, label, source

    def __len__(self):
        return len(self.ds)
    

# command to retrieve dataset from cached files
@dataset.command
def get_file_cached_dataset(name, audio_processor, cache_root_path):
    print("get_file_cached_dataset::", name, audio_processor['identifier'], "sr=", audio_processor['sr'],
          cache_root_path)
    ds = FilesCachedDataset(ContentDataset, name, audio_processor['identifier'], cache_root_path)
    return ds


# commands to create the datasets for training, testing and evaluation
@dataset.command
def get_training_set_raw():
    ds = get_base_training_set_raw()
    return ds


@dataset.command
def get_base_training_set_raw(meta_json,  train_files_json, subsample, use_full_dev_dataset):
    train_files = pd.read_json(train_files_json).transpose()['note_str'].values.reshape(-1)
    meta = pd.read_json(meta_json).transpose()
    meta.index = [i for i in range(len(meta))]
    train_indices = list(meta[meta['note_str'].isin(train_files)].index)
    if subsample:
        train_indices = np.random.choice(train_indices, size=subsample, replace=False)
    if use_full_dev_dataset:
        train_indices = np.arange(len(meta['note_str']))
    ds = SimpleSelectionDataset(get_file_cached_dataset(), train_indices)
    return ds


@dataset.command
def get_test_set_raw():
    ds = get_base_test_set_raw()
    return ds


@dataset.command
def get_base_test_set_raw(meta_json, test_files_json, subsample):
    test_files = pd.read_json(test_files_json).transpose()['note_str'].values.reshape(-1)
    meta = pd.read_json(meta_json).transpose()
    meta.index = [i for i in range(len(meta))]
    test_indices = list(meta[meta['note_str'].isin(test_files)].index)
    if subsample:
        test_indices = np.random.choice(test_indices, size=subsample, replace=False)
    ds = SimpleSelectionDataset(get_file_cached_dataset(), test_indices)
    return ds
