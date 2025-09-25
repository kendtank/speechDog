# -*- coding: utf-8 -*-
"""
@Time    : 2025/09/22
@Author  : Kend
@FileName: preprocess_dog_barks.py
@Software: PyCharm
@Description: 预处理狗吠数据集 → 统一长度 Mel/MFCC 特征 + 数据增强
"""

import os
import numpy as np
import librosa
import random
from tqdm import tqdm


# --------------------
# 参数配置
# --------------------
INPUT_DIR = r"D:\kend\myPython\speechDog-master\datasets\compare_dog"  # 原始狗吠 wav
OUTPUT_DIR = r"D:\kend\myPython\speechDog-master\datasets\dog_embedding_features"  # 输出特征目录
SAMPLE_RATE = 16000
TARGET_LEN = 0.4       # 目标时长，秒 (增加到0.4秒以捕获更多狗吠特征)
N_MELS = 40            # log-mel bin 数
N_MFCC = 13            # MFCC 基础系数数目
HOP_LENGTH = 160       # 10ms (16kHz)
N_FFT = 400            # 25ms 窗
AUG_PER_SAMPLE = 8     # 增加增强次数到8次以提高数据多样性

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)

# --------------------
# 工具函数
# --------------------
def load_and_pad(file, target_len=TARGET_LEN, sr=SAMPLE_RATE):
    """加载音频并 pad/crop 到目标长度（RMS 定位能量峰值）"""
    y, orig_sr = librosa.load(file, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)

    target_samples = int(target_len * sr)

    if len(y) < target_samples:
        pad_len = target_samples - len(y)
        left = random.randint(0, pad_len)
        right = pad_len - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_samples:
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
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
    aug_data.append(y)  # 原始
    
    # 1. 加噪声 (白噪声)
    noise = np.random.normal(0, 0.005, len(y))
    aug_data.append(y + noise)
    
    # 2. 音量扰动
    aug_data.append(y * random.uniform(0.8, 1.2))
    
    # 3. 时间拉伸
    rate = random.uniform(0.9, 1.1)
    try:
        stretched = librosa.effects.time_stretch(y, rate=rate)
        # 确保长度一致
        if len(stretched) > len(y):
            stretched = stretched[:len(y)]
        else:
            pad_len = len(y) - len(stretched)
            left = pad_len // 2
            right = pad_len - left
            stretched = np.pad(stretched, (left, right), mode="constant")
        aug_data.append(stretched)
    except:
        print("Time stretch failed")
        # 如果时间拉伸失败，使用原始音频
        aug_data.append(y)
    
    # 4. 轻微音高偏移（±0.5 半音）
    n_steps = random.uniform(-0.5, 0.5)
    try:
        pitched = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
        aug_data.append(pitched)
    except:
        print(f"Pitch shift failed")
        # 如果音高偏移失败，使用原始音频
        aug_data.append(y)
    
    # 5. 随机静音插入
    aug_y = y.copy()
    start = random.randint(0, len(y) - int(0.05 * sr))
    aug_y[start:start + int(0.05 * sr)] = 0
    aug_data.append(aug_y)
    
    # 6. 动态范围压缩
    compressed = np.tanh(y * 2) / 2
    aug_data.append(compressed)
    
    # 7. 频率遮蔽 (模拟部分频率丢失)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mask_freq = random.randint(0, D.shape[0] - 10)
    D[mask_freq:mask_freq+5, :] = 0  # 遮蔽5个频率bin
    masked_y = librosa.istft(D, hop_length=HOP_LENGTH)
    # 确保长度一致
    if len(masked_y) > len(y):
        masked_y = masked_y[:len(y)]
    elif len(masked_y) < len(y):
        pad_len = len(y) - len(masked_y)
        left = pad_len // 2
        right = pad_len - left
        masked_y = np.pad(masked_y, (left, right), mode="constant")
    aug_data.append(masked_y)

    return aug_data

def wav_to_logmel(y, sr=SAMPLE_RATE, n_mels=N_MELS):
    """转换为 log-mel 特征"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)  # per-sample 归一化
    return mel_db.astype(np.float32)

def wav_to_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """转换为 MFCC 特征 (13+Δ+ΔΔ=39)"""
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, delta, delta2])
    feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-6)
    return feat.astype(np.float32)

def extract_spectral_features(y, sr=SAMPLE_RATE):
    """提取额外的频谱特征以增强识别能力"""
    # 谱质心
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # 零交叉率
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # 谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # 谱滚降点 (roll-off)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # 将特征组合成一个向量
    features = np.vstack([spectral_centroids, zcr, spectral_bandwidth, spectral_rolloff])
    features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-6)
    return features.astype(np.float32)

# --------------------
# 主流程
# --------------------
def process_dataset():
    for dog_id in tqdm(os.listdir(INPUT_DIR), desc="Processing dogs"):
        dog_path = os.path.join(INPUT_DIR, dog_id)
        if not os.path.isdir(dog_path):
            continue

        save_dir_mel = os.path.join(OUTPUT_DIR, "logmel", dog_id)
        save_dir_mfcc = os.path.join(OUTPUT_DIR, "mfcc", dog_id)
        save_dir_spectral = os.path.join(OUTPUT_DIR, "spectral", dog_id)  # 新增频谱特征目录
        os.makedirs(save_dir_mel, exist_ok=True)
        os.makedirs(save_dir_mfcc, exist_ok=True)
        os.makedirs(save_dir_spectral, exist_ok=True)  # 创建新目录

        for fname in os.listdir(dog_path):
            if not fname.lower().endswith(".wav"):
                continue

            wav_path = os.path.join(dog_path, fname)
            y = load_and_pad(wav_path)

            aug_versions = augment(y)
            for i, aug_y in enumerate(aug_versions):
                mel = wav_to_logmel(aug_y)
                mfcc = wav_to_mfcc(aug_y)
                spectral = extract_spectral_features(aug_y)  # 提取额外的频谱特征

                base = os.path.splitext(fname)[0]
                np.save(os.path.join(save_dir_mel, f"{base}_aug{i}.npy"), mel.astype(np.float16))
                np.save(os.path.join(save_dir_mfcc, f"{base}_aug{i}.npy"), mfcc.astype(np.float16))
                np.save(os.path.join(save_dir_spectral, f"{base}_aug{i}.npy"), spectral.astype(np.float16))  # 保存频谱特征

    print(f"数据预处理完成，保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()


"""
logmel目录：
存储log-mel频谱特征
这是一种常用的音频特征表示方法，能够较好地模拟人耳听觉特性
适用于传统的机器学习分类器
mfcc目录：
存储MFCC（Mel频率倒谱系数）特征，包括基础MFCC以及Δ和ΔΔ特征
MFCC是语音和音频识别中广泛使用的特征，能够有效表示音频的频谱包络
对于声纹识别任务非常有效
spectral目录：
存储额外的频谱特征，包括谱质心、零交叉率、谱带宽和谱滚降点
这些特征提供了关于音频信号不同方面的信息，有助于更好地区分不同的狗叫声
这些目录中的特征数据可以用于训练和测试不同的狗吠声纹识别模型。每个目录下会按照狗的ID进一步分类存储，便于后续的训练和验证过程。
"""