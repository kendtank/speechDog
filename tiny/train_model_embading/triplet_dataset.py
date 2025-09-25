# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/22 下午3:23
@Author  : Kend + GPT
@FileName: triplet_dataset.py
@Software: PyCharm
@modifier: Triplet Dataset for Dog Bark Voiceprint Verification
"""



import os
import random
import numpy as np
import tensorflow as tf

class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=32, input_type="logmel"):
        """
        root_dir: 数据集根目录 (如 logmel/)
                  logmel/dog01/*.npy, logmel/dog02/*.npy ...
        batch_size: 每个 batch 多少 triplets
        input_type: "logmel" / "mfcc" / "spectral"
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.input_type = input_type

        # 加载数据
        self.data = self._load_data()
        self.classes = list(self.data.keys())

    def _load_data(self):
        """读取目录下的 .npy 特征，按类别组织"""
        data = {}
        for dog_id in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir, dog_id)
            if not os.path.isdir(path):
                continue
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")]
            if files:
                data[dog_id] = files
        return data

    def __len__(self):
        return 1000  # 每个 epoch 迭代多少个 batch，可调整

    def __getitem__(self, idx):
        anchors, positives, negatives = [], [], []

        for _ in range(self.batch_size):
            # 1. 随机选一个类别作为 Anchor/Positive
            pos_class = random.choice(self.classes)
            neg_class = random.choice([c for c in self.classes if c != pos_class])

            # 2. 在 pos_class 里取两条不同样本
            a_file, p_file = random.sample(self.data[pos_class], 2)
            # 3. 在 neg_class 里取一条样本
            n_file = random.choice(self.data[neg_class])

            # 加载特征
            a = np.load(a_file)
            p = np.load(p_file)
            n = np.load(n_file)

            anchors.append(a)
            positives.append(p)
            negatives.append(n)

        # 转换成 Tensor 格式
        anchors = np.expand_dims(np.array(anchors), -1)  # (B, H, W, 1)
        positives = np.expand_dims(np.array(positives), -1)
        negatives = np.expand_dims(np.array(negatives), -1)

        return [anchors, positives, negatives], None
