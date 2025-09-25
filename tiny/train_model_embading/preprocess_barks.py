# -*- coding: utf-8 -*-
"""
@Time    : 2025/09/20
@Author  : Kend
@FileName: preprocess_barks.py
@Software: PyCharm
@Description: 预处理狗吠数据集 → 统一长度 Mel 频谱 + 数据增强
"""

import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import random
from tqdm import tqdm

# --------------------
# 参数配置
# --------------------
INPUT_DIR = r"/datasets/compare_dog"  # 原始裁剪后的狗吠 wav
OUTPUT_DIR = r"/datasets/compare_dog_mel_dataset"  # 处理后的 mel 数据
SAMPLE_RATE = 16000
TARGET_LEN = 0.3       # 目标时长，单位秒（建议 0.3）根据狗吠的长度来设置， 一般0.3-0.5秒
N_MELS = 40            # Mel bin 数
HOP_LENGTH = 160       # 10ms (16kHz)
N_FFT = 400            # 25ms 窗
AUG_PER_SAMPLE = 5     # 每个原始音频增强次数

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------
# 工具函数
# --------------------

def load_and_pad(file, target_len=TARGET_LEN, sr=SAMPLE_RATE):
    """加载音频并 pad/crop 到目标长度"""
    y, orig_sr = librosa.load(file, sr=None, mono=True)  # 保持原始 sr
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr) # 显式重采样

    target_samples = int(target_len * sr)

    if len(y) < target_samples:
        # 短音频 → 前后随机 pad
        pad_len = target_samples - len(y)
        left = random.randint(0, pad_len)
        right = pad_len - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_samples:
        # 长音频 → 裁剪能量最大的部分
        rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
        max_frame = np.argmax(rms)
        center = int(max_frame * HOP_LENGTH + N_FFT // 2)
        start = max(0, center - target_samples // 2)
        end = start + target_samples
        if end > len(y):
            end = len(y)
            start = end - target_samples
        y = y[start:end]

    return y



def augment(y, sr=SAMPLE_RATE):
    """数据增强：返回多个版本"""
    aug_data = []
    # 1. 原始
    aug_data.append(y)
    # 2. 加噪声
    noise = np.random.normal(0, 0.005, len(y))
    aug_data.append(y + noise)
    # 3. 音量扰动()
    aug_data.append(y * random.uniform(0.8, 1.2))
    # 4. 时间拉伸
    rate = random.uniform(0.9, 1.1)
    aug_data.append(librosa.effects.time_stretch(y, rate=rate)[:len(y)])
    # 5. 音高偏移
    n_steps = random.choice([-1, 1])
    aug_data.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps))
    return aug_data


def wav_to_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS):
    """转换为 Mel 频谱"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


# --------------------
# 主流程
# --------------------
def process_dataset():
    for root, dirs, files in os.walk(INPUT_DIR):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            dog_id = os.path.basename(root)  # dog01 / dog02
            save_dir = os.path.join(OUTPUT_DIR, dog_id)
            os.makedirs(save_dir, exist_ok=True)

            wav_path = os.path.join(root, fname)
            y = load_and_pad(wav_path)

            # 数据增强
            aug_versions = augment(y)
            for i, aug_y in enumerate(aug_versions):
                mel = wav_to_mel(aug_y)
                npy_path = os.path.join(
                    save_dir, f"{os.path.splitext(fname)[0]}_aug{i}.npy"
                )
                np.save(npy_path, mel)

    print(f"✅ 数据预处理完成，保存到 {OUTPUT_DIR}")



if __name__ == "__main__":
    process_dataset()
