# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午2:07
@Author  : Kend
@FileName: dataset.py
@Software: PyCharm
@modifier:
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random


class DogVocalDataset(Dataset):
    def __init__(self, data_root, dog_ids, transform=None):
        self.data_root = data_root
        self.dog_to_label = {dog: idx for idx, dog in enumerate(dog_ids)}
        self.samples = []  # [(path, label), ...]

        for dog in dog_ids:
            dog_dir = os.path.join(data_root, dog)
            for f in os.listdir(dog_dir):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(dog_dir, f), self.dog_to_label[dog]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path).astype(np.float32)  # (32, 32)
        mel = np.expand_dims(mel, axis=0)  # (1, 32, 32) for CNN
        return torch.tensor(mel), label


def create_train_val_datasets(data_root, val_ratio=0.2, seed=42):
    dog_ids = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    random.seed(seed)
    np.random.seed(seed)

    # 按狗划分：每只狗的部分样本用于验证（模拟真实验证场景）
    train_samples, val_samples = [], []
    for dog in dog_ids:
        dog_dir = os.path.join(data_root, dog)
        all_files = [f for f in os.listdir(dog_dir) if f.endswith('.npy')]
        # 分离原始和增强样本
        orig_files = [f for f in all_files if '_orig.npy' in f]
        aug_files = [f for f in all_files if '_aug' in f]

        # 验证集：只用原始样本（更真实）
        n_val = max(1, int(len(orig_files) * val_ratio))
        val_orig = orig_files[:n_val]
        train_orig = orig_files[n_val:]

        # 训练集 = 剩余原始 + 所有增强
        train_files = train_orig + aug_files

        for f in train_files:
            train_samples.append((os.path.join(dog_dir, f), dog))
        for f in val_orig:
            val_samples.append((os.path.join(dog_dir, f), dog))

    # 创建数据集
    train_dogs = list(set([dog for _, dog in train_samples]))
    val_dogs = list(set([dog for _, dog in val_samples]))

    # 保存路径-标签对
    train_paths_labels = [(p, train_dogs.index(dog)) for p, dog in train_samples]
    val_paths_labels = [(p, val_dogs.index(dog)) for p, dog in val_samples]

    class SimpleDataset(Dataset):
        def __init__(self, paths_labels):
            self.paths_labels = paths_labels

        def __len__(self):
            return len(self.paths_labels)

        def __getitem__(self, idx):
            path, label = self.paths_labels[idx]
            mel = np.load(path).astype(np.float32)
            mel = np.expand_dims(mel, 0)
            return torch.tensor(mel), label

    return SimpleDataset(train_paths_labels), SimpleDataset(val_paths_labels), train_dogs, val_dogs