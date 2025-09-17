# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/17 下午5:30
@Author  : Kend
@FileName: compare_audio_processor.py
@Software: PyCharm
@modifier: Qwen3 coder
"""

"""
增强版音频处理器，根据用户提供的方向实现更精细的音频处理策略：

1. 改进滤波策略：
   - 多级带通/组合滤波
   - 自适应滤波

2. 时域去噪：
   - 噪声门（Noise Gate）
   - 谱减法（Spectral Subtraction）
   - 瞬态增强

3. 频谱+特征选择：
   - 只保留狗吠特征区间

4. 模板 vs 推理：
   - 模板阶段（干净）
   - 推理阶段（宽松）
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

def adaptive_bandpass(y, sr, low_percentile=10, high_percentile=90):
    """
    自适应带通滤波，根据音频能量谱动态确定低/高频边界
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        low_percentile (int): 低频截止百分位数
        high_percentile (int): 高频截止百分位数
    
    Returns:
        np.array: 滤波后的音频信号
    """
    # 计算短时傅里叶变换
    D = librosa.stft(y)
    # 计算功率谱密度
    psd = np.abs(D)**2
    
    # 计算频率轴
    freqs = librosa.fft_frequencies(sr=sr)
    
    # 计算平均功率谱
    mean_psd = np.mean(psd, axis=1)
    
    # 计算累积分布
    cumulative = np.cumsum(mean_psd)
    cumulative = cumulative / cumulative[-1]
    
    # 根据百分位数确定边界
    low_idx = np.argmax(cumulative >= low_percentile / 100.0)
    high_idx = np.argmax(cumulative >= high_percentile / 100.0)
    
    low_freq = max(freqs[low_idx], 100)   # 最低不低于100Hz
    high_freq = min(freqs[high_idx], sr/2 - 100)  # 最高不高于奈奎斯特频率-100Hz
    
    print(f"[自适应滤波] 检测到主要频段: {low_freq:.0f}Hz - {high_freq:.0f}Hz")
    
    # 应用带通滤波
    return bandpass(y, sr, low=low_freq, high=high_freq, order=4)

def multi_stage_filter(y, sr):
    """
    多级滤波处理
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        np.array: 多级滤波后的音频信号
    """
    # 第一级：粗滤波，去除明显低频/高频杂音
    y_coarse = bandpass(y, sr, low=150, high=8000, order=4)
    
    # 第二级：细滤波，对狗吠的典型频段做窄带滤波
    y_fine = bandpass(y_coarse, sr, low=300, high=7500, order=6)
    
    return y_fine

def noise_gate(y, sr, threshold_db=-30, attack_time=0.01, release_time=0.05):
    """
    噪声门处理，对低于门限的音量直接抑制
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        threshold_db (float): 门限(dB)
        attack_time (float): 攻击时间(秒)
        release_time (float): 释放时间(秒)
    
    Returns:
        np.array: 噪声门处理后的音频信号
    """
    # 转换为dB
    threshold_amp = 10**(threshold_db/20)
    
    # 计算包络
    envelope = np.abs(sps.hilbert(y))
    
    # 计算攻击和释放系数
    attack_coeff = np.exp(-np.log(9) / (attack_time * sr))
    release_coeff = np.exp(-np.log(9) / (release_time * sr))
    
    # 应用噪声门
    gain = np.zeros_like(envelope)
    current_gain = 0.0
    
    for i in range(len(envelope)):
        if envelope[i] > current_gain:
            current_gain = attack_coeff * current_gain + (1 - attack_coeff) * envelope[i]
        else:
            current_gain = release_coeff * current_gain + (1 - release_coeff) * envelope[i]
        
        gain[i] = 1.0 if current_gain > threshold_amp else 0.0
    
    return y * gain

def spectral_subtraction(y, sr, noise_factor=0.1):
    """
    谱减法去噪，估计背景噪声谱，再从整个音频谱中减去
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        noise_factor (float): 噪声估计因子
    
    Returns:
        np.array: 谱减法处理后的音频信号
    """
    # 估计背景噪声（使用前0.1秒）
    noise_duration = min(int(0.1 * sr), len(y))
    noise_segment = y[:noise_duration]
    
    # 计算噪声谱
    noise_stft = librosa.stft(noise_segment)
    noise_psd = np.mean(np.abs(noise_stft)**2, axis=1)
    
    # 计算整个信号的STFT
    D = librosa.stft(y)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 谱减法
    magnitude_squared = magnitude**2
    subtracted = magnitude_squared - noise_factor * noise_psd[:, np.newaxis]
    subtracted = np.maximum(subtracted, 0.01 * magnitude_squared)  # 避免负数
    magnitude_clean = np.sqrt(subtracted)
    
    # 重构信号
    D_clean = magnitude_clean * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean)
    
    return y_clean

def transient_enhancement(y, sr, enhancement_factor=2.0):
    """
    瞬态增强，对突发尖锐声音增强
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        enhancement_factor (float): 增强因子
    
    Returns:
        np.array: 瞬态增强后的音频信号
    """
    # 计算短时能量
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)
    
    # 使用短时傅里叶变换计算能量
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    energy = np.sum(np.abs(D)**2, axis=0)
    
    # 计算能量的移动平均作为背景
    window_size = 5
    energy_background = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
    
    # 计算瞬态指标
    transient_indicator = np.maximum(energy / (energy_background + 1e-8), 1.0)
    
    # 应用增强
    enhanced_frames = []
    for i in range(D.shape[1]):
        enhancement = 1.0 + (enhancement_factor - 1.0) * min(transient_indicator[i] / 5.0, 1.0)
        enhanced_frames.append(D[:, i] * enhancement)
    
    # 重构信号
    D_enhanced = np.column_stack(enhanced_frames)
    y_enhanced = librosa.istft(D_enhanced, hop_length=hop_length)
    
    # 保持原始信号长度
    if len(y_enhanced) > len(y):
        y_enhanced = y_enhanced[:len(y)]
    elif len(y_enhanced) < len(y):
        y_enhanced = np.pad(y_enhanced, (0, len(y) - len(y_enhanced)))
    
    return y_enhanced

def process_for_template_enhanced(y, sr):
    """
    模板阶段增强处理，极致干净
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        np.array: 处理后的音频信号
    """
    print("[模板阶段] 开始增强处理...")
    
    # 1. 多级滤波
    y_filtered = multi_stage_filter(y, sr)
    print("[模板阶段] 多级滤波完成")
    
    # 2. 自适应滤波
    y_adaptive = adaptive_bandpass(y_filtered, sr)
    print("[模板阶段] 自适应滤波完成")
    
    # 3. 噪声门
    y_gate = noise_gate(y_adaptive, sr, threshold_db=-35)
    print("[模板阶段] 噪声门处理完成")
    
    # 4. 谱减法
    y_spectral = spectral_subtraction(y_gate, sr, noise_factor=0.15)
    print("[模板阶段] 谱减法处理完成")
    
    # 5. 瞬态增强
    y_enhanced = transient_enhancement(y_spectral, sr, enhancement_factor=1.5)
    print("[模板阶段] 瞬态增强完成")
    
    return y_enhanced

def process_for_inference_simple(y, sr):
    """
    推理阶段简单处理，保留大部分声音
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        np.array: 处理后的音频信号
    """
    print("[推理阶段] 开始简单处理...")
    
    # 只做低阶带通
    y_filtered = bandpass(y, sr, low=150, high=8000, order=4)
    print("[推理阶段] 带通滤波完成")
    
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
        y_processed = process_for_template_enhanced(y, sr)
    else:  # inference
        y_processed = process_for_inference_simple(y, sr)
    
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
        output_dir = os.path.join(base_dir, f"processed_{mode}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    mode_suffix = "clean" if mode == "template" else "wide"
    output_path = os.path.join(output_dir, f"{base_name}_{mode_suffix}.wav")
    
    sf.write(output_path, y, sr)
    print(f"已保存处理后的音频到: {output_path}")
    
    return output_path

# 示例用法
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python compare_audio_processor.py <音频文件路径> [模式]")
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
python detect_similar_sounds/tools/compare_audio_processor.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\outdoor_braking.mp3 template

# 推理阶段处理
python detect_similar_sounds/tools/compare_audio_processor.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\outdoor_braking.mp3 inference


结果：能够有效的去噪音和别的无关的声音， 但是会影响狗吠本体的声音
"""