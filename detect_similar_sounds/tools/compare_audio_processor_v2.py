# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/17 下午8:30
@Author  : Kend
@FileName: compare_audio_processor_v2.py
@Software: PyCharm
@modifier: Qwen3 coder
"""

"""
特征保护型音频处理器，专注于在去除噪声的同时保护狗吠的主体特征：

1. 精确噪声检测与去除
2. 保护狗吠主体特征
3. 最小化信号失真
"""

import os
import sys

# 添加项目根目录到Python路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
import numpy as np
import librosa
import scipy.signal as sps
import soundfile as sf
from detect_similar_sounds.bark_segmentation import bandpass


def detect_bark_frequency_range(y, sr):
    """
    检测狗吠的主要频率范围

    Args:
        y (np.array): 音频信号
        sr (int): 采样率

    Returns:
        tuple: (low_freq, high_freq) 狗吠的主要频率范围
    """
    # 计算短时傅里叶变换
    D = librosa.stft(y)
    # 计算功率谱密度
    psd = np.abs(D) ** 2

    # 计算频率轴
    freqs = librosa.fft_frequencies(sr=sr)

    # 计算平均功率谱
    mean_psd = np.mean(psd, axis=1)

    # 找到能量最高的频率点作为参考
    peak_freq_idx = np.argmax(mean_psd)
    peak_freq = freqs[peak_freq_idx]

    # 狗吠的典型频率范围通常在 300Hz - 8000Hz 之间
    # 但我们会根据检测到的峰值进行调整
    low_freq = max(peak_freq * 0.3, 300)  # 不低于300Hz
    high_freq = min(peak_freq * 3.0, 8000)  # 不高于8000Hz

    print(f"[频率检测] 狗吠主要频率范围: {low_freq:.0f}Hz - {high_freq:.0f}Hz")

    return int(low_freq), int(high_freq)


def selective_noise_reduction(y, sr, noise_factor=0.1):
    """
    选择性噪声抑制，在保护狗吠特征的同时去除背景噪声

    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        noise_factor (float): 噪声估计因子（较小值以保护狗吠特征）

    Returns:
        np.array: 噪声抑制后的音频信号
    """
    print("[噪声抑制] 开始选择性噪声抑制...")

    # 估计背景噪声（使用前0.1秒）
    noise_duration = min(int(0.1 * sr), len(y))
    noise_segment = y[:noise_duration]

    # 计算噪声谱
    noise_stft = librosa.stft(noise_segment)
    noise_psd = np.mean(np.abs(noise_stft) ** 2, axis=1)

    # 计算整个信号的STFT
    D = librosa.stft(y)
    magnitude = np.abs(D)
    phase = np.angle(D)

    # 检测狗吠的主要频率范围
    bark_low, bark_high = detect_bark_frequency_range(y, sr)

    # 计算频率轴
    freqs = librosa.fft_frequencies(sr=sr)

    # 创建频率掩码，保护狗吠频率范围
    bark_mask = (freqs >= bark_low) & (freqs <= bark_high)

    # 对非狗吠频率范围应用更强的噪声抑制
    magnitude_squared = magnitude ** 2
    for i, freq in enumerate(freqs):
        if not bark_mask[i]:  # 非狗吠频率范围
            subtracted = magnitude_squared[i] - noise_factor * 2.0 * noise_psd[i]
            magnitude_squared[i] = np.maximum(subtracted, 0.1 * magnitude_squared[i])
        else:  # 狗吠频率范围
            subtracted = magnitude_squared[i] - noise_factor * 0.5 * noise_psd[i]
            magnitude_squared[i] = np.maximum(subtracted, 0.5 * magnitude_squared[i])

    magnitude_clean = np.sqrt(magnitude_squared)

    # 重构信号
    D_clean = magnitude_clean * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean)

    print("[噪声抑制] 选择性噪声抑制完成")
    return y_clean


def preserve_bark_features(y, sr):
    """
    保护狗吠特征的处理

    Args:
        y (np.array): 音频信号
        sr (int): 采样率

    Returns:
        np.array: 特征保护处理后的音频信号
    """
    print("[特征保护] 开始保护狗吠特征...")

    # 使用温和的带通滤波器，保护狗吠频率范围
    bark_low, bark_high = detect_bark_frequency_range(y, sr)
    y_filtered = bandpass(y, sr, low=max(bark_low - 100, 150), high=min(bark_high + 1000, 8000), order=4)

    print("[特征保护] 狗吠特征保护完成")
    return y_filtered


def soft_noise_gate(y, sr, threshold_db=-25, attack_time=0.01, release_time=0.1):
    """
    软噪声门，在保护狗吠特征的同时适度抑制背景噪声

    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        threshold_db (float): 门限(dB)
        attack_time (float): 攻击时间(秒)
        release_time (float): 释放时间(秒)

    Returns:
        np.array: 软噪声门处理后的音频信号
    """
    print("[软噪声门] 开始软噪声门处理...")

    # 转换为dB
    threshold_amp = 10 ** (threshold_db / 20)

    # 计算包络
    envelope = np.abs(sps.hilbert(y))

    # 计算攻击和释放系数
    attack_coeff = np.exp(-np.log(9) / (attack_time * sr))
    release_coeff = np.exp(-np.log(9) / (release_time * sr))

    # 应用软噪声门
    gain = np.zeros_like(envelope)
    current_gain = 0.0

    for i in range(len(envelope)):
        if envelope[i] > current_gain:
            current_gain = attack_coeff * current_gain + (1 - attack_coeff) * envelope[i]
        else:
            current_gain = release_coeff * current_gain + (1 - release_coeff) * envelope[i]

        # 使用软过渡而不是硬切
        if current_gain > threshold_amp:
            gain[i] = 1.0  # 完全保留
        else:
            # 软过渡：根据距离阈值的距离确定保留程度
            ratio = current_gain / threshold_amp
            gain[i] = 0.3 + 0.7 * ratio  # 最少保留30%

    y_gated = y * gain
    print("[软噪声门] 软噪声门处理完成")
    return y_gated


def process_for_template_preserving(y, sr):
    """
    模板阶段特征保护处理，专注于保护狗吠主体特征

    Args:
        y (np.array): 音频信号
        sr (int): 采样率

    Returns:
        np.array: 处理后的音频信号
    """
    print("[模板阶段] 开始特征保护处理...")

    # 1. 保护狗吠特征
    y_preserved = preserve_bark_features(y, sr)
    print("[模板阶段] 狗吠特征保护完成")

    # 2. 选择性噪声抑制
    y_denoised = selective_noise_reduction(y_preserved, sr, noise_factor=0.08)
    print("[模板阶段] 选择性噪声抑制完成")

    # 3. 软噪声门处理
    y_gated = soft_noise_gate(y_denoised, sr, threshold_db=-25)
    print("[模板阶段] 软噪声门处理完成")

    return y_gated


def process_for_inference_preserving(y, sr):
    """
    推理阶段特征保护处理，保留大部分声音特征

    Args:
        y (np.array): 音频信号
        sr (int): 采样率

    Returns:
        np.array: 处理后的音频信号
    """
    print("[推理阶段] 开始特征保护处理...")

    # 只做温和的带通滤波
    y_filtered = bandpass(y, sr, low=200, high=8000, order=4)
    print("[推理阶段] 温和带通滤波完成")

    return y_filtered


def process_audio_file(audio_path, mode='template'):
    """
    处理单个音频文件

    Args:
        audio_path (str): 音频文件路径
        mode (str): 处理模式 'template' 或 'inference'

    Returns:
        np.array: 处理后的音频信号
        int: 采样率
    """
    print(f"处理音频文件: {audio_path}, 模式: {mode}")

    # 读取音频
    y, sr = librosa.load(audio_path, sr=16000)

    # 根据模式选择处理方法
    if mode == 'template':
        y_processed = process_for_template_preserving(y, sr)
    else:  # inference
        y_processed = process_for_inference_preserving(y, sr)

    return y_processed, sr


def save_processed_audio(y, sr, original_path, mode, output_dir=None):
    """
    保存处理后的音频

    Args:
        y (np.array): 处理后的音频信号
        sr (int): 采样率
        original_path (str): 原始音频文件路径
        mode (str): 处理模式
        output_dir (str): 输出目录

    Returns:
        str: 保存的文件路径
    """
    if output_dir is None:
        base_dir = os.path.dirname(original_path)
        output_dir = os.path.join(base_dir, f"processed_{mode}_preserving")

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_path))[0]
    mode_suffix = "clean_preserving" if mode == "template" else "wide_preserving"
    output_path = os.path.join(output_dir, f"{base_name}_{mode_suffix}.wav")

    sf.write(output_path, y, sr)
    print(f"已保存处理后的音频到: {output_path}")

    return output_path


# 示例用法
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python compare_audio_processor_v2.py <音频文件路径> [模式]")
        print("模式: 'template' 或 'inference'，默认为 'template'")
        sys.exit(1)

    audio_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'template'

    # 处理音频
    y_processed, sr = process_audio_file(audio_path, mode)

    # 保存结果
    save_processed_audio(y_processed, sr, audio_path, mode)

"""
# 模板阶段处理（默认）
python detect_similar_sounds/tools/compare_audio_processor_v2.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\outdoor_braking.mp3 template

# 推理阶段处理
python detect_similar_sounds/tools/compare_audio_processor_v2.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\outdoor_braking.mp3 inference

"""