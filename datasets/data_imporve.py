# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 下午4:12
@Author  : Kend
@FileName: data_imporve.py
@Software: PyCharm
@modifier:
"""



import os
import glob
import random
import shutil
import numpy as np
import librosa
import soundfile as sf

# ========== 配置 ==========
input_dir = r"D:\work\datasets\tinyML\bark_16k"  # 输入原始音频
output_dir = "output"  # 输出目录
target_num = 300   # 目标总数量
train_num = 200
val_num = 100
sr = 16000  # 统一采样率

# ========== 工具函数 ==========

def band_energy_ratio(y, sr, fmin=300, fmax=8000):
    """计算频带能量占比"""
    S = np.abs(librosa.stft(y))**2
    freqs = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(S)
    band_energy = np.sum(S[(freqs >= fmin) & (freqs <= fmax), :])
    if total_energy == 0:
        return 0
    return band_energy / total_energy

def check_valid(y, sr):
    """检查主体完整度 & 时长"""
    # 时长
    dur = librosa.get_duration(y=y, sr=sr)
    if dur < 0.1:
        return None
    if dur > 1.0:
        y = y[:int(sr * 1.0)]  # 截断到 1s

    # 主体完整度
    ratio = band_energy_ratio(y, sr)
    if ratio < 0.95:
        return None

    return y

def augment_audio(y, sr):
    """轻量增广"""
    choice = random.choice(["volume", "shift", "noise"])
    if choice == "volume":
        factor = np.random.uniform(0.9, 1.1)  # 不超过 ±10%
        y = y * factor
    elif choice == "shift":
        shift = np.random.randint(-int(0.02 * sr), int(0.02 * sr))  # ±20ms
        y = np.roll(y, shift)
    elif choice == "noise":
        noise = np.random.normal(0, 0.003, len(y))  # 轻微噪声
        y = y + noise
    return y

# ========== 主流程 ==========
def main():
    all_files = glob.glob(os.path.join(input_dir, "*.wav"))
    print("找到的音频文件数量:", len(all_files))
    if not all_files:
        raise ValueError("没有找到音频文件，请检查路径！")

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "all")
    os.makedirs(temp_dir, exist_ok=True)

    augmented_files = []
    idx = 0

    while len(augmented_files) < target_num:
        file = random.choice(all_files)
        y, _ = librosa.load(file, sr=sr, mono=True)

        # 检查并修正
        y = check_valid(y, sr)
        if y is None:
            continue

        if len(augmented_files) < len(all_files):
            new_y = y  # 原始
        else:
            new_y = augment_audio(y, sr)  # 增广
            new_y = check_valid(new_y, sr)
            if new_y is None:
                continue

        save_path = os.path.join(temp_dir, f"aug_{idx}.wav")
        sf.write(save_path, new_y, sr, subtype="PCM_16")
        augmented_files.append(save_path)

        # 打印信息
        dur = librosa.get_duration(y=new_y, sr=sr)
        ratio = band_energy_ratio(new_y, sr)
        print(f"[{idx}] {os.path.basename(save_path)} | 时长={dur:.3f}s | 主体完整度={ratio:.3f}")

        idx += 1

    print(f"✅ 最终生成音频数量: {len(augmented_files)}")

    # 打乱
    random.shuffle(augmented_files)

    # 划分
    train_files = augmented_files[:train_num]
    val_files = augmented_files[train_num:train_num + val_num]

    # 保存
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for f in train_files:
        shutil.copy(f, train_dir)
    for f in val_files:
        shutil.copy(f, val_dir)

    print(f"训练集: {len(train_files)}，验证集: {len(val_files)}")
    print(f"数据已保存到: {output_dir}")

if __name__ == "__main__":
    main()
