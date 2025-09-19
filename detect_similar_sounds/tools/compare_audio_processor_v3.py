# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/17 下午8:30
@Author  : Kend
@FileName: compare_audio_processor_v3.py
@Software: PyCharm
@modifier: Qwen3 coder
"""

"""
特征保护型音频处理器，专注于在去除噪声的同时保护狗吠的主体特征：

1. 精确噪声检测与去除
2. 保护狗吠主体特征
3. 最小化信号失真

优化版本：
- 添加了自适应噪声估计
- 改进了狗吠频率范围检测
- 增加了特征保护的动态阈值
- 引入了高级频谱处理技术
- 添加了处理质量评估
- 增加了MFCC特征保护机制
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
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import deque

def detect_bark_frequency_range(y, sr):
    """
    改进的狗吠频率范围检测
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        tuple: (low_freq, high_freq) 狗吠的主要频率范围
    """
    # 计算短时傅里叶变换，使用重叠窗口提高精度
    n_fft = 2048 if sr >= 16000 else 1024
    hop_length = n_fft // 4
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # 计算功率谱密度并应用平滑滤波
    psd = np.abs(D)**2
    psd_smoothed = gaussian_filter1d(psd, sigma=1, axis=0)
    
    # 计算频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 计算平均功率谱和阈值
    mean_psd = np.mean(psd_smoothed, axis=1)
    
    # 找到显著能量区域（超过平均能量的一定比例）
    energy_threshold = np.max(mean_psd) * 0.3  # 使用更高比例以找到真正的能量集中区域
    significant_freqs = freqs[mean_psd >= energy_threshold]
    
    # 找出多个峰值区域，使用滑动窗口方法
    peak_regions = []
    window_size = 5
    if len(significant_freqs) > window_size:
        for i in range(len(significant_freqs) - window_size + 1):
            window = significant_freqs[i:i+window_size]
            window_mean = np.mean(window)
            peak_regions.append(window_mean)
        
        # 对峰值区域进行聚类
        if peak_regions:
            # 简化聚类：找到最集中的频率区域
            median_freq = np.median(np.array(peak_regions))
        else:
            median_freq = 1000  # 默认值
    else:
        median_freq = 1000  # 默认值
    
    # 结合先验知识和动态检测结果
    # 狗吠的典型频率范围通常在 200Hz - 8000Hz 之间
    if len(significant_freqs) > 0:
        # 使用实际检测到的范围，但限制在合理范围内
        detected_low = np.min(significant_freqs)
        detected_high = np.max(significant_freqs)
        
        # 扩展频率范围以确保覆盖完整的狗吠特征
        low_freq = max(detected_low * 0.8, 200)   # 向下扩展20%
        high_freq = min(detected_high * 1.3, 8000) # 向上扩展30%
    else:
        # 如果没有检测到显著频率，使用基于中位数的动态范围
        low_freq = max(median_freq * 0.5, 200)
        high_freq = min(median_freq * 2.5, 8000)
    
    # 确保频率范围有足够宽度
    min_bandwidth = 500  # 最小带宽500Hz
    if high_freq - low_freq < min_bandwidth:
        center_freq = (low_freq + high_freq) / 2
        half_band = min_bandwidth / 2
        low_freq = max(center_freq - half_band, 200)
        high_freq = min(center_freq + half_band, 8000)
    
    print(f"[频率检测] 狗吠主要频率范围: {low_freq:.0f}Hz - {high_freq:.0f}Hz")
    
    return int(low_freq), int(high_freq)

def adaptive_noise_estimation(y, sr, frame_length=25, hop_length=10):
    """
    自适应噪声估计，能够更准确地估计非语音段的噪声特性
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        frame_length (int): 帧长度(ms)
        hop_length (int): 帧移(ms)
    
    Returns:
        np.array: 估计的噪声功率谱密度
    """
    # 转换为样本数
    frame_size = int(frame_length * sr / 1000)
    hop_size = int(hop_length * sr / 1000)
    
    # 计算每帧能量
    frame_energy = []
    for i in range(0, len(y) - frame_size, hop_size):
        frame = y[i:i+frame_size]
        energy = np.sum(frame**2) / frame_size
        frame_energy.append(energy)
    
    # 基于能量排序，取能量最低的20%作为噪声样本
    sorted_energy = sorted(frame_energy)
    noise_threshold_idx = int(len(sorted_energy) * 0.2)
    noise_threshold = sorted_energy[noise_threshold_idx]
    
    # 收集噪声帧并计算噪声谱
    n_fft = 2048 if sr >= 16000 else 1024
    noise_frames = []
    for i in range(0, len(y) - frame_size, hop_size):
        if frame_energy[i//hop_size] <= noise_threshold:
            frame = y[i:i+frame_size]
            noise_frames.append(frame)
    
    # 如果噪声帧不足，使用前0.1秒作为噪声
    if len(noise_frames) < 5:
        noise_duration = min(int(0.1 * sr), len(y))
        noise_segment = y[:noise_duration]
        noise_stft = librosa.stft(noise_segment, n_fft=n_fft)
    else:
        # 合并噪声帧
        noise_segment = np.concatenate(noise_frames)
        noise_stft = librosa.stft(noise_segment, n_fft=n_fft)
    
    # 计算噪声功率谱密度
    noise_psd = np.mean(np.abs(noise_stft)**2, axis=1)
    
    return noise_psd

def selective_noise_reduction(y, sr, noise_factor=0.05):
    """
    改进的选择性噪声抑制，自适应保护狗吠特征
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        noise_factor (float): 噪声估计因子（较小值以保护狗吠特征）
    
    Returns:
        np.array: 噪声抑制后的音频信号
    """
    print("[噪声抑制] 开始选择性噪声抑制...")
    
    # 使用自适应噪声估计
    noise_psd = adaptive_noise_estimation(y, sr)
    
    # 计算整个信号的STFT
    n_fft = 2048 if sr >= 16000 else 1024
    hop_length = n_fft // 4
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 检测狗吠的主要频率范围
    bark_low, bark_high = detect_bark_frequency_range(y, sr)
    
    # 计算频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 创建频率掩码，保护狗吠频率范围
    bark_mask = (freqs >= bark_low) & (freqs <= bark_high)
    
    # 调整噪声抑制参数，动态保护狗吠特征
    magnitude_squared = magnitude**2
    for i in range(magnitude_squared.shape[0]):
        # 频率轴保护
        freq_protection = 1.0
        if bark_mask[i]:
            # 在狗吠频率范围内，增强保护
            freq_protection = 0.2
        else:
            # 在非狗吠频率范围内，适度抑制
            freq_protection = 1.0
        
        # 对每个频率点应用自适应噪声抑制
        for j in range(magnitude_squared.shape[1]):
            # 信号能量归一化，自适应调整抑制强度
            signal_energy = magnitude_squared[i, j]
            
            # 动态阈值：信号越强，抑制越弱
            if signal_energy > noise_factor * 10 * noise_psd[i]:
                # 强信号，几乎不抑制
                suppression_factor = 0.1
            elif signal_energy > noise_factor * 3 * noise_psd[i]:
                # 中等信号，轻度抑制
                suppression_factor = 0.5
            else:
                # 弱信号，较强抑制但保留一定能量
                suppression_factor = 1.0
            
            # 结合频率保护和信号能量的最终抑制因子
            final_factor = noise_factor * freq_protection * suppression_factor
            
            # 应用抑制
            subtracted = signal_energy - final_factor * noise_psd[i]
            # 确保不会过度抑制（保留至少原始信号的10%）
            magnitude_squared[i, j] = np.maximum(subtracted, 0.1 * signal_energy)
    
    # 使用频谱平滑，进一步减少噪声和失真
    magnitude_squared = gaussian_filter1d(magnitude_squared, sigma=1, axis=0)
    magnitude_squared = gaussian_filter1d(magnitude_squared, sigma=0.5, axis=1)
    
    # 确保非负
    magnitude_squared = np.maximum(magnitude_squared, 0.0001)
    magnitude_clean = np.sqrt(magnitude_squared)
    
    # 重构信号
    D_clean = magnitude_clean * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=hop_length)
    
    print("[噪声抑制] 选择性噪声抑制完成")
    return y_clean

def extract_mfcc_features(y, sr):
    """
    提取MFCC特征，用于特征保护评估
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        np.array: MFCC特征矩阵
    """
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    # 添加一阶和二阶差分特征
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    # 合并特征
    combined_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    
    return combined_features

def preserve_bark_features(y, sr):
    """
    改进的狗吠特征保护处理
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        np.array: 特征保护处理后的音频信号
    """
    print("[特征保护] 开始保护狗吠特征...")
    
    # 检测狗吠的主要频率范围，使用更宽的频率范围以捕获更多特征
    bark_low, bark_high = detect_bark_frequency_range(y, sr)
    bark_low_expanded = max(bark_low - 100, 100)  # 扩展低频范围
    bark_high_expanded = min(bark_high + 1000, 10000)  # 扩展高频范围
    
    # 提取原始信号的MFCC特征，用于后续评估
    original_mfcc = extract_mfcc_features(y, sr)
    
    # 使用多级带通滤波器策略
    # 1. 主带通滤波器 - 使用扩展的频率范围
    y_filtered = bandpass(y, sr, low=bark_low_expanded, high=bark_high_expanded, order=4)
    
    # 2. 动态增益控制 - 保护信号的动态范围
    # 计算信号包络
    envelope = np.abs(sps.hilbert(y_filtered))
    # 平滑包络
    smoothed_envelope = gaussian_filter1d(envelope, sigma=sr//1500)  # 更平滑的滤波
    # 归一化包络
    envelope_norm = smoothed_envelope / (np.max(smoothed_envelope) + 1e-10)
    
    # 动态增益调整：更温和的增益曲线，减少过度增强
    gain_factor = np.where(envelope_norm < 0.2, 1.3 - 1.3 * envelope_norm, 1.0)
    y_dynamic = y_filtered * gain_factor
    
    # 3. 信号谐波保护 - 改进的谐波增强策略
    # 检测信号中的谐波结构并增强
    n_fft = 2048 if sr >= 16000 else 1024
    hop_length = n_fft // 4
    D = librosa.stft(y_dynamic, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 谐波增强
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bark_mask = (freqs >= bark_low) & (freqs <= bark_high)
    
    # 对狗吠频率范围内的谐波结构进行增强 - 使用自适应增益
    for i in range(magnitude.shape[1]):  # 对每一帧
        # 找到当前帧的主要能量集中区域
        max_energy_freq = freqs[np.argmax(magnitude[:, i])]
        
        # 只对狗吠频率范围内的帧进行谐波增强
        if bark_low <= max_energy_freq <= bark_high:
            # 使用更智能的谐波增强策略：基于峰值附近的能量分布
            for j in range(len(freqs)):
                if bark_mask[j] and magnitude[j, i] > 0.3 * np.max(magnitude[:, i]):
                    # 对强信号谐波使用更温和的增强
                    magnitude[j, i] *= 1.1  # 从1.2降低到1.1，减少过度增强
    
    # 重构信号
    D_enhanced = magnitude * np.exp(1j * phase)
    y_harmonic = librosa.istft(D_enhanced, hop_length=hop_length)
    
    # 4. 特征保护评估 - 确保MFCC特征变化最小
    processed_mfcc = extract_mfcc_features(y_harmonic, sr)
    
    # 计算特征差异
    feature_diff = np.mean(np.abs(original_mfcc - processed_mfcc))
    print(f"[特征保护] MFCC特征变化: {feature_diff:.4f}")
    
    # 如果特征变化过大，混合原始信号进行补偿
    if feature_diff > 0.4:  # 降低阈值，更早进行补偿
        print("[特征保护] 特征变化较大，应用信号补偿...")
        # 确保两个信号长度一致
        max_length = max(len(y_harmonic), len(y_filtered))
        y_harmonic_fixed = librosa.util.fix_length(y_harmonic, size=max_length)
        y_filtered_fixed = librosa.util.fix_length(y_filtered, size=max_length)
        y_compensated = 0.6 * y_harmonic_fixed + 0.4 * y_filtered_fixed  # 增加原始滤波信号的比例
    else:
        y_compensated = y_harmonic
    
    print("[特征保护] 狗吠特征保护完成")
    return y_compensated

def adaptive_soft_noise_gate(y, sr, base_threshold_db=-25, attack_time=0.01, release_time=0.1):
    """
    自适应软噪声门，根据信号特性动态调整参数
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        base_threshold_db (float): 基础门限(dB)
        attack_time (float): 攻击时间(秒)
        release_time (float): 释放时间(秒)
    
    Returns:
        np.array: 噪声门处理后的音频信号
    """
    print("[软噪声门] 开始自适应软噪声门处理...")
    
    # 计算信号的RMS和峰值以自适应设置门限
    signal_rms = np.sqrt(np.mean(y**2))
    signal_peak = np.max(np.abs(y))
    
    # 动态调整门限：根据信号能量调整
    if signal_rms > 0.01:
        # 强信号，适当降低门限以保留更多细节
        threshold_db = base_threshold_db - 5
    else:
        # 弱信号，保持原始门限
        threshold_db = base_threshold_db
    
    # 转换为振幅
    threshold_amp = 10**(threshold_db/20)
    
    # 计算平滑包络，使用更智能的包络检测方法
    # 1. 使用希尔伯特变换计算瞬时幅度
    analytic_signal = sps.hilbert(y)
    instantaneous_amplitude = np.abs(analytic_signal)
    
    # 2. 应用平滑滤波，减少快速波动
    window_size = int(0.05 * sr)  # 50ms窗口
    if window_size % 2 == 0:
        window_size += 1  # 确保窗口大小为奇数
    
    # 计算中值滤波的kernel_size，确保为奇数
    kernel_size = window_size // 5
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保kernel_size为奇数
    if kernel_size < 3:
        kernel_size = 3  # 确保kernel_size至少为3
    
    # 使用中值滤波和高斯平滑结合的方法
    envelope_median = sps.medfilt(instantaneous_amplitude, kernel_size=kernel_size)
    envelope_smoothed = gaussian_filter1d(envelope_median, sigma=window_size//2)
    
    # 计算攻击和释放系数
    attack_coeff = np.exp(-np.log(9) / (attack_time * sr))
    release_coeff = np.exp(-np.log(9) / (release_time * sr))
    
    # 应用自适应噪声门
    gain = np.zeros_like(envelope_smoothed)
    current_gain = 0.0
    
    # 使用滑动窗口检测信号活动区域
    activity_window = deque(maxlen=int(0.1 * sr))  # 100ms活动检测窗口
    
    for i in range(len(envelope_smoothed)):
        # 检测信号活动
        is_active = envelope_smoothed[i] > threshold_amp
        activity_window.append(is_active)
        
        # 计算当前活动级别
        activity_level = sum(activity_window) / len(activity_window)
        
        # 根据活动级别动态调整释放时间
        dynamic_release_time = release_time * (1.0 - 0.5 * activity_level)
        dynamic_release_coeff = np.exp(-np.log(9) / (dynamic_release_time * sr))
        
        # 平滑增益变化
        if envelope_smoothed[i] > current_gain:
            current_gain = attack_coeff * current_gain + (1 - attack_coeff) * envelope_smoothed[i]
        else:
            current_gain = dynamic_release_coeff * current_gain + (1 - dynamic_release_coeff) * envelope_smoothed[i]
        
        # 自适应增益控制：更精细的软过渡
        if current_gain > threshold_amp:
            # 信号段：完全保留或根据信号强度适度调整
            if current_gain > 2 * threshold_amp:
                gain[i] = 1.0  # 强信号完全保留
            else:
                # 中等信号：渐进增益
                gain[i] = 0.8 + 0.2 * ((current_gain - threshold_amp) / threshold_amp)
        else:
            # 噪声段：适度抑制但保留一定背景
            ratio = current_gain / threshold_amp
            # 根据噪声级别调整抑制程度
            if ratio < 0.3:
                gain[i] = 0.1  # 极低信号，强烈抑制
            elif ratio < 0.7:
                gain[i] = 0.2 + 0.4 * ratio  # 中等抑制
            else:
                gain[i] = 0.5 + 0.5 * ratio  # 轻度抑制
    
    # 应用增益
    y_gated = y * gain
    
    # 防止增益过高导致的信号削波
    max_amplitude = np.max(np.abs(y_gated))
    if max_amplitude > 0.95:
        y_gated = y_gated * 0.95 / max_amplitude
        print("[软噪声门] 检测到削波风险，应用限幅...")
    
    print("[软噪声门] 自适应软噪声门处理完成")
    return y_gated

def soft_noise_gate(y, sr, threshold_db=-25, attack_time=0.01, release_time=0.1):
    """
    软噪声门，调用自适应实现
    
    保留原函数接口以保持兼容性
    """
    return adaptive_soft_noise_gate(y, sr, base_threshold_db=threshold_db, 
                                    attack_time=attack_time, release_time=release_time)

def evaluate_processing_quality(original_y, processed_y, sr):
    """
    评估音频处理质量，比较原始信号和处理后信号的特征差异
    
    Args:
        original_y (np.array): 原始音频信号
        processed_y (np.array): 处理后的音频信号
        sr (int): 采样率
    
    Returns:
        dict: 包含各种质量评估指标的字典
    """
    # 确保两个信号长度相同
    min_len = min(len(original_y), len(processed_y))
    original_y = original_y[:min_len]
    processed_y = processed_y[:min_len]
    
    # 计算信号能量保留率
    original_energy = np.sum(original_y**2)
    processed_energy = np.sum(processed_y**2)
    energy_ratio = processed_energy / original_energy if original_energy > 0 else 0
    
    # 计算信噪比变化
    # 估计原始信号中的噪声
    original_noise = original_y - processed_y
    snr_improvement = 10 * np.log10(original_energy / (np.sum(original_noise**2) + 1e-10)) if original_energy > 0 else 0
    
    # 计算特征相似度
    original_mfcc = extract_mfcc_features(original_y, sr)
    processed_mfcc = extract_mfcc_features(processed_y, sr)
    
    # 确保MFCC特征长度相同
    min_mfcc_len = min(original_mfcc.shape[1], processed_mfcc.shape[1])
    original_mfcc = original_mfcc[:, :min_mfcc_len]
    processed_mfcc = processed_mfcc[:, :min_mfcc_len]
    
    # 计算MFCC特征的余弦相似度
    mfcc_flat_original = original_mfcc.flatten()
    mfcc_flat_processed = processed_mfcc.flatten()
    
    # 归一化特征向量
    mfcc_flat_original = mfcc_flat_original / (np.linalg.norm(mfcc_flat_original) + 1e-10)
    mfcc_flat_processed = mfcc_flat_processed / (np.linalg.norm(mfcc_flat_processed) + 1e-10)
    
    feature_similarity = np.dot(mfcc_flat_original, mfcc_flat_processed)
    
    # 返回评估结果
    quality_metrics = {
        'energy_ratio': energy_ratio,
        'snr_improvement_db': snr_improvement,
        'feature_similarity': feature_similarity
    }
    
    print(f"[质量评估] 能量保留率: {energy_ratio:.2f}, SNR改善: {snr_improvement:.2f}dB, 特征相似度: {feature_similarity:.4f}")
    
    return quality_metrics

def process_for_template_preserving(y, sr):
    """
    改进的模板阶段特征保护处理，专注于保护狗吠主体特征
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        np.array: 处理后的音频信号
    """
    print("[模板阶段] 开始特征保护处理...")
    
    # 保存原始信号用于质量评估和特征恢复
    original_y = np.copy(y)
    
    # 1. 轻微预加重处理，减少高频衰减
    pre_emphasis = 0.95
    y_pre_emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    print("[模板阶段] 预加重处理完成")
    
    # 2. 保护狗吠特征
    y_preserved = preserve_bark_features(y_pre_emphasized, sr)
    print("[模板阶段] 狗吠特征保护完成")
    
    # 3. 自适应选择性噪声抑制，使用温和的参数保护特征
    y_denoised = selective_noise_reduction(y_preserved, sr, noise_factor=0.03)
    print("[模板阶段] 选择性噪声抑制完成")
    
    # 4. 自适应软噪声门处理，使用更适合模板创建的参数
    y_gated = adaptive_soft_noise_gate(y_denoised, sr, base_threshold_db=-30, attack_time=0.02, release_time=0.15)
    print("[模板阶段] 自适应软噪声门处理完成")
    
    # 5. 信号归一化，保持一致的音量水平
    max_amplitude = np.max(np.abs(y_gated))
    if max_amplitude > 0:
        y_normalized = y_gated * 0.9 / max_amplitude
    else:
        y_normalized = y_gated
    print("[模板阶段] 信号归一化完成")
    
    # 6. 评估处理质量
    quality_metrics = evaluate_processing_quality(original_y, y_normalized, sr)
    
    # 7. 特征保护检查：如果特征相似度低于阈值，混合原始和特征保护信号
    if quality_metrics['feature_similarity'] < 0.9:
        print("[模板阶段] 特征相似度较低，应用特征恢复...")
        # 确保两个信号长度一致
        max_length = max(len(y_normalized), len(y_preserved))
        y_normalized_fixed = librosa.util.fix_length(y_normalized, size=max_length)
        y_preserved_fixed = librosa.util.fix_length(y_preserved, size=max_length)
        y_enhanced = 0.7 * y_normalized_fixed + 0.3 * y_preserved_fixed
    else:
        y_enhanced = y_normalized
        
    # 8. 最终特征恢复：添加一个额外的特征恢复步骤，即使相似度较高
    # 确保信号长度一致
    y_enhanced_fixed = librosa.util.fix_length(y_enhanced, size=len(original_y))
    # 轻微混合原始信号以保留更多原始特征
    y_final = 0.95 * y_enhanced_fixed + 0.05 * original_y
    
    # 重新归一化
    max_amplitude = np.max(np.abs(y_final))
    if max_amplitude > 0:
        y_final = y_final * 0.9 / max_amplitude
    
    print("[模板阶段] 特征保护处理完成")
    return y_final

def process_for_inference_preserving(y, sr, preserve_features=True):
    """
    改进的推理阶段特征保护处理，保留狗吠特征同时提高实时性能
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        preserve_features (bool): 是否启用高级特征保护（默认True）
    
    Returns:
        np.array: 处理后的音频信号
    """
    print("[推理阶段] 开始特征保护处理...")
    
    # 保存原始信号用于质量评估
    original_y = np.copy(y)
    
    if preserve_features:
        # 完整特征保护模式
        print("[推理阶段] 启用完整特征保护模式...")
        
        # 1. 温和的预加重，保留高频特征
        pre_emphasis = 0.95
        y_pre_emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        print("[推理阶段] 预加重处理完成")
        
        # 2. 自适应带通滤波，基于检测的狗吠频率
        bark_low, bark_high = detect_bark_frequency_range(y_pre_emphasized, sr)
        # 对于推理阶段，使用稍宽的频率范围以确保捕获所有可能的狗吠变体
        y_filtered = bandpass(y_pre_emphasized, sr, 
                             low=max(bark_low-100, 100), 
                             high=min(bark_high+1000, 10000), 
                             order=4)
        print("[推理阶段] 自适应带通滤波完成")
        
        # 3. 轻量级噪声抑制，使用较高的噪声因子以保护更多细节
        y_denoised = selective_noise_reduction(y_filtered, sr, noise_factor=0.08)
        print("[推理阶段] 轻量级噪声抑制完成")
        
        # 4. 自适应噪声门，使用更低的门限以保留更多信号
        y_gated = adaptive_soft_noise_gate(y_denoised, sr, base_threshold_db=-35, 
                                          attack_time=0.01, release_time=0.1)
        print("[推理阶段] 自适应噪声门处理完成")
        
        # 5. 信号归一化
        max_amplitude = np.max(np.abs(y_gated))
        if max_amplitude > 0:
            y_normalized = y_gated * 0.8 / max_amplitude
        else:
            y_normalized = y_gated
        print("[推理阶段] 信号归一化完成")
        
        # 6. 评估处理质量
        quality_metrics = evaluate_processing_quality(original_y, y_normalized, sr)
        
        # 7. 如果需要，应用简单的特征恢复
        if quality_metrics['feature_similarity'] < 0.85:
            print("[推理阶段] 特征相似度较低，应用轻量级特征恢复...")
            # 确保两个信号长度一致
            max_length = max(len(y_normalized), len(y_filtered))
            y_normalized_fixed = librosa.util.fix_length(y_normalized, size=max_length)
            y_filtered_fixed = librosa.util.fix_length(y_filtered, size=max_length)
            y_processed = 0.9 * y_normalized_fixed + 0.1 * y_filtered_fixed
        else:
            y_processed = y_normalized
    else:
        # 快速处理模式（低延迟）
        print("[推理阶段] 启用快速处理模式（低延迟）...")
        
        # 只进行必要的处理以保持低延迟
        # 1. 简单的带通滤波
        y_filtered = bandpass(y, sr, low=200, high=8000, order=2)
        print("[推理阶段] 快速带通滤波完成")
        
        # 2. 基本的噪声门处理
        y_gated = soft_noise_gate(y_filtered, sr, threshold_db=-30)
        print("[推理阶段] 基本噪声门处理完成")
        
        y_processed = y_gated
    
    print("[推理阶段] 特征保护处理完成")
    return y_processed

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
python detect_similar_sounds/tools/compare_audio_processor_v3.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\outdoor_braking.mp3 template

# 推理阶段处理
python detect_similar_sounds/tools/compare_audio_processor_v3.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\outdoor_braking.mp3 inference

"""