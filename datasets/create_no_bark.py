# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 下午4:03
@Author  : Kend
@FileName: create_no_bark.py
@Software: PyCharm
@modifier:
"""


import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import random
from collections import defaultdict

# 配置
esc_csv = r"D:\work\datasets\ESC-50-master\meta\esc50.csv"  # ESC50 标签文件
esc_audio_dir = r"D:\work\datasets\ESC-50-master\audio"  # ESC50 音频目录
output_root = "./neg_samples_balanced"
train_dir = os.path.join(output_root, "train")
val_dir = os.path.join(output_root, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

min_len = 0.3
max_len = 0.8
sample_rate = 16000

target_total = 600  # 总共要生成的负样本
train_ratio = 2/3   # 训练集比例

# 读取 csv
df = pd.read_csv(esc_csv)

# 非狗类
non_dog_df = df[df["category"] != "dog"]
categories = sorted(non_dog_df["category"].unique())

# 平均分配数量
per_class_target = target_total // len(categories)  # 每类采样数
print(f"每类目标数: {per_class_target}, 类别数: {len(categories)}")

# 按类别分配
samples_collected = defaultdict(list)

for category in categories:
    class_df = non_dog_df[non_dog_df["category"] == category]
    class_files = class_df["filename"].tolist()

    # 每类裁剪片段直到达到 per_class_target
    count = 0
    while count < per_class_target:
        fname = random.choice(class_files)
        filepath = os.path.join(esc_audio_dir, fname)

        try:
            y, sr = librosa.load(filepath, sr=sample_rate, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < min_len:
                continue

            seg_len = random.uniform(min_len, max_len)
            if duration <= seg_len:
                continue

            start_time = random.uniform(0, duration - seg_len)
            start_sample = int(start_time * sr)
            end_sample = int((start_time + seg_len) * sr)
            y_seg = y[start_sample:end_sample]

            # 保存
            seg_id = len(samples_collected[category])
            out_name = f"{category}_{os.path.splitext(fname)[0]}_seg{seg_id}.wav"

            # 临时先存列表，后面统一分配 train/val
            samples_collected[category].append((y_seg, sr, out_name))

            count += 1

        except Exception as e:
            print(f"Error processing {fname}: {e}")

# ==================
# 统一划分 train/val
# ==================
for category, samples in samples_collected.items():
    random.shuffle(samples)
    n_total = len(samples)
    n_train = int(n_total * train_ratio)

    for i, (y_seg, sr, out_name) in enumerate(samples):
        if i < n_train:
            out_path = os.path.join(train_dir, out_name)
        else:
            out_path = os.path.join(val_dir, out_name)
        sf.write(out_path, y_seg, sr)

print("负样本收集完成 ✅")
print(f"训练集: {len(os.listdir(train_dir))}, 验证集: {len(os.listdir(val_dir))}")
