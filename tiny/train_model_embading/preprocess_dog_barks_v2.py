# -*- coding: utf-8 -*-
"""
@Time    : 2025/09/22
@Author  : Kend
@FileName: preprocess_dog_barks_v2.py
@Software: PyCharm
@Description: 预处理狗吠数据集 → 统一长度 Mel/MFCC 特征 + 数据增强
"""

import os
import numpy as np
import librosa
import random
from tqdm import tqdm

# --------------------
# 参数配置（面向 tinyML 验证）
# --------------------
INPUT_DIR = r"D:\kend\myPython\speechDog-master\datasets\compare_dog"
OUTPUT_DIR = r"D:\kend\myPython\speechDog-master\datasets\dog_tiny_verification"

SAMPLE_RATE = 16000
TARGET_DURATION = 0.4  # 秒
N_MELS = 32  # 降低到 32，匹配 32x32 输入
N_FFT = 400  # 25ms
HOP_LENGTH = 200  # 12.5ms → 0.4s / 0.0125s ≈ 32 帧
TARGET_TIME_FRAMES = 32  # 固定时间帧数

AUG_PER_SAMPLE = 6  # 减少到 6 次，移除高风险增强

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)


# --------------------
# 工具函数
# --------------------
def load_and_crop_to_peak(file, target_duration=TARGET_DURATION, sr=SAMPLE_RATE):
    """加载并以能量峰值为中心裁剪到固定时长"""
    y, orig_sr = librosa.load(file, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)

    target_samples = int(target_duration * sr)

    if len(y) < target_samples:
        # 补零居中
        pad_len = target_samples - len(y)
        left = pad_len // 2
        right = pad_len - left
        y = np.pad(y, (left, right), mode="constant")
    else:
        # 用 RMS 找峰值帧
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        if len(rms) == 0:
            center = len(y) // 2
        else:
            max_frame = np.argmax(rms)
            center = int(max_frame * HOP_LENGTH + N_FFT // 2)
        start = max(0, center - target_samples // 2)
        end = start + target_samples
        if end > len(y):
            end = len(y)
            start = max(0, end - target_samples)
        y = y[start:end]
    return y[:target_samples]  # 确保长度严格一致


def augment(y, sr=SAMPLE_RATE):
    """保留对声纹鲁棒性有益的增强"""
    aug_list = [y]  # 原始样本（用于注册）

    # 1. 加白噪声（低强度）
    noise = np.random.normal(0, 0.003, len(y))
    aug_list.append(y + noise)

    # 2. 音量扰动
    aug_list.append(y * random.uniform(0.85, 1.15))

    # 3. 轻微时间拉伸（±10%）
    rate = random.uniform(0.92, 1.08)
    try:
        stretched = librosa.effects.time_stretch(y, rate=rate)
        if len(stretched) > len(y):
            stretched = stretched[:len(y)]
        else:
            stretched = np.pad(stretched, (0, len(y) - len(stretched)), mode='constant')
        aug_list.append(stretched)
    except:
        aug_list.append(y)

    # 4. 轻微音高偏移（±0.3 半音）
    n_steps = random.uniform(-0.3, 0.3)
    try:
        pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        aug_list.append(pitched)
    except:
        aug_list.append(y)

    # 5. 边缘静音（仅开头或结尾，10ms）
    aug_y = y.copy()
    if random.random() > 0.5:
        start = 0
        length = int(0.01 * sr)
    else:
        start = len(y) - int(0.01 * sr)
        length = int(0.01 * sr)
    aug_y[start:start + length] = 0
    aug_list.append(aug_y)

    return aug_list


def extract_logmel_fixed(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_time=TARGET_TIME_FRAMES):
    """提取 log-mel 并强制 resize 到 (n_mels, n_time)"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # per-sample normalization
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # 时间轴插值到固定长度
    if mel_db.shape[1] > n_time:
        mel_db = mel_db[:, :n_time]
    else:
        mel_db = librosa.util.fix_length(mel_db, size=n_time, axis=1)

    return mel_db.astype(np.float32)


# --------------------
# 主流程
# --------------------
def process_dataset():
    for dog_id in tqdm(os.listdir(INPUT_DIR), desc="Processing dogs"):
        dog_path = os.path.join(INPUT_DIR, dog_id)
        if not os.path.isdir(dog_path):
            continue

        output_dir = os.path.join(OUTPUT_DIR, dog_id)
        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(dog_path):
            if not fname.lower().endswith(".wav"):
                continue

            wav_path = os.path.join(dog_path, fname)
            try:
                y = load_and_crop_to_peak(wav_path)
            except Exception as e:
                print(f"Failed to load {wav_path}: {e}")
                continue

            # 生成增强版本（第一个是原始）
            aug_versions = augment(y)

            base = os.path.splitext(fname)[0]
            for i, aug_y in enumerate(aug_versions):
                if i >= AUG_PER_SAMPLE + 1:  # 原始 + AUG_PER_SAMPLE 个增强
                    break
                mel = extract_logmel_fixed(aug_y)
                # 标记是否为原始样本（可用于注册）
                is_original = (i == 0)
                suffix = "_orig" if is_original else f"_aug{i}"
                np.save(
                    os.path.join(output_dir, f"{base}{suffix}.npy"),
                    mel.astype(np.float16)
                )

    print(f"✅ 预处理完成！输出目录: {OUTPUT_DIR}")
    print("💡 提示：训练时可使用所有样本；注册模板请仅使用 *_orig.npy 文件")


if __name__ == "__main__":
    process_dataset()