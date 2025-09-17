# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/17 下午9:30
@Author  : Kend
@FileName: comparison_visualizer.py
@Software: PyCharm
@modifier: Qwen3 coder
"""

"""
音频对比可视化工具，用于比较降噪前后的音频并检测狗吠特征损失
"""

import os
import sys
# 添加项目根目录到Python路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.spatial.distance import cosine
from detect_similar_sounds.bark_segmentation import bandpass

# 设置中文字体以避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    psd = np.abs(D)**2
    
    # 计算频率轴
    freqs = librosa.fft_frequencies(sr=sr)
    
    # 计算平均功率谱
    mean_psd = np.mean(psd, axis=1)
    
    # 找到能量最高的频率点作为参考
    peak_freq_idx = np.argmax(mean_psd)
    peak_freq = freqs[peak_freq_idx]
    
    # 狗吠的典型频率范围通常在 300Hz - 8000Hz 之间
    # 但我们会根据检测到的峰值进行调整
    low_freq = max(peak_freq * 0.3, 300)   # 不低于300Hz
    high_freq = min(peak_freq * 3.0, 8000) # 不高于8000Hz
    
    return int(low_freq), int(high_freq)

def compute_bark_features(y, sr):
    """
    计算狗吠声的特征，用于比较降噪前后的特征损失
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        dict: 包含各种声学特征的字典
    """
    # 检测狗吠的主要频率范围
    bark_low, bark_high = detect_bark_frequency_range(y, sr)
    
    # 应用带通滤波，只保狗吠频率范围内的信号
    y_filtered = bandpass(y, sr, low=bark_low, high=bark_high, order=4)
    
    # 提取MFCC特征（仅在狗吠频率范围内）
    mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 提取谱质心（仅在狗吠频率范围内）
    spectral_centroids = librosa.feature.spectral_centroid(y=y_filtered, sr=sr)[0]
    centroid_mean = np.mean(spectral_centroids)
    centroid_std = np.std(spectral_centroids)
    
    # 提取零交叉率
    zcr = librosa.feature.zero_crossing_rate(y)[0]  # 原始信号的ZCR
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # 提取谱带宽（仅在狗吠频率范围内）
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr)[0]
    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_std = np.std(spectral_bandwidth)
    
    # 提取谱滚降点（仅在狗吠频率范围内）
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_filtered, sr=sr)[0]
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_std = np.std(spectral_rolloff)
    
    # 提取RMS能量
    rms = librosa.feature.rms(y=y_filtered)[0]  # 过滤后信号的RMS
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    return {
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
        'centroid_mean': centroid_mean,
        'centroid_std': centroid_std,
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std,
        'bandwidth_mean': bandwidth_mean,
        'bandwidth_std': bandwidth_std,
        'rolloff_mean': rolloff_mean,
        'rolloff_std': rolloff_std,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'bark_freq_range': (bark_low, bark_high)
    }

def compare_bark_features(features_orig, features_proc):
    """
    比较原始和处理后的狗吠特征，计算损失
    
    Args:
        features_orig (dict): 原始音频特征
        features_proc (dict): 处理后音频特征
    
    Returns:
        dict: 特征损失度量
    """
    losses = {}
    
    # 计算MFCC损失（使用余弦距离）
    mfcc_mean_loss = cosine(features_orig['mfcc_mean'], features_proc['mfcc_mean'])
    mfcc_std_loss = cosine(features_orig['mfcc_std'], features_proc['mfcc_std'])
    
    losses['mfcc_mean_loss'] = mfcc_mean_loss
    losses['mfcc_std_loss'] = mfcc_std_loss
    
    # 计算其他特征的相对变化
    for key in ['centroid_mean', 'zcr_mean', 'bandwidth_mean', 'rolloff_mean', 'rms_mean']:
        orig_val = features_orig[key]
        proc_val = features_proc[key]
        # 相对变化百分比
        rel_change = abs(proc_val - orig_val) / (orig_val + 1e-8) * 100
        losses[f'{key}_change_pct'] = rel_change
    
    # 计算总体相似度评分（越接近0越好）
    total_loss = (mfcc_mean_loss + mfcc_std_loss) / 2
    for key in ['centroid_mean', 'zcr_mean', 'bandwidth_mean', 'rolloff_mean', 'rms_mean']:
        total_loss += losses[f'{key}_change_pct'] / 100.0
    
    losses['total_feature_loss'] = total_loss / 6  # 平均损失
    
    return losses

def visualize_audio_comparison(y_orig, y_proc, sr, title="音频对比"):
    """
    可视化原始和处理后的音频对比
    
    Args:
        y_orig (np.array): 原始音频信号
        y_proc (np.array): 处理后音频信号
        sr (int): 采样率
        title (str): 图形标题
    """
    plt.figure(figsize=(15, 12))
    
    # 1. 波形对比
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y_orig, sr=sr, alpha=0.6, label="原始")
    librosa.display.waveshow(y_proc, sr=sr, alpha=0.8, label="处理后")
    plt.title("波形对比")
    plt.legend()
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    
    # 2. 频谱对比
    plt.subplot(3, 2, 2)
    fft_orig = np.abs(np.fft.rfft(y_orig))
    fft_proc = np.abs(np.fft.rfft(y_proc))
    freqs = np.fft.rfftfreq(len(y_orig), 1 / sr)
    
    plt.semilogy(freqs, fft_orig, alpha=0.7, label="原始")
    plt.semilogy(freqs, fft_proc, alpha=0.7, label="处理后")
    plt.title("频谱对比 (对数尺度)")
    plt.xlabel("频率 (Hz)")
    plt.ylabel("幅度")
    plt.legend()
    
    # 3. MFCC特征对比
    plt.subplot(3, 2, 3)
    mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr, n_mfcc=13)
    mfcc_proc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=13)
    
    plt.imshow(mfcc_orig, aspect='auto', origin='lower')
    plt.title("原始音频 MFCC")
    plt.xlabel("帧")
    plt.ylabel("MFCC系数")
    
    plt.subplot(3, 2, 4)
    plt.imshow(mfcc_proc, aspect='auto', origin='lower')
    plt.title("处理后音频 MFCC")
    plt.xlabel("帧")
    plt.ylabel("MFCC系数")
    
    # 4. 谱质心对比
    plt.subplot(3, 2, 5)
    centroid_orig = librosa.feature.spectral_centroid(y=y_orig, sr=sr)[0]
    centroid_proc = librosa.feature.spectral_centroid(y=y_proc, sr=sr)[0]
    frames = range(len(centroid_orig))
    times = librosa.frames_to_time(frames, sr=sr)
    
    plt.plot(times, centroid_orig, alpha=0.7, label="原始")
    plt.plot(times, centroid_proc, alpha=0.7, label="处理后")
    plt.title("谱质心对比")
    plt.xlabel("时间 (秒)")
    plt.ylabel("频率 (Hz)")
    plt.legend()
    
    # 5. RMS能量对比
    plt.subplot(3, 2, 6)
    rms_orig = librosa.feature.rms(y=y_orig)[0]
    rms_proc = librosa.feature.rms(y=y_proc)[0]
    frames = range(len(rms_orig))
    times = librosa.frames_to_time(frames, sr=sr)
    
    plt.plot(times, rms_orig, alpha=0.7, label="原始")
    plt.plot(times, rms_proc, alpha=0.7, label="处理后")
    plt.title("RMS能量对比")
    plt.xlabel("时间 (秒)")
    plt.ylabel("能量")
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_and_visualize_audio_loss(audio_path_orig, audio_path_proc=None, y_proc=None, sr_proc=None):
    """
    分析和可视化音频处理的特征损失
    
    Args:
        audio_path_orig (str): 原始音频文件路径
        audio_path_proc (str): 处理后音频文件路径（可选）
        y_proc (np.array): 处理后的音频信号（可选，如果已提供则不需要audio_path_proc）
        sr_proc (int): 处理后音频的采样率（可选，如果已提供y_proc则需要）
    """
    # 加载原始音频
    y_orig, sr_orig = librosa.load(audio_path_orig, sr=16000)
    
    # 获取处理后的音频
    if y_proc is not None and sr_proc is not None:
        # 直接使用提供的处理后音频
        if sr_proc != sr_orig:
            # 如果采样率不同，需要重采样
            y_proc = librosa.resample(y_proc, orig_sr=sr_proc, target_sr=sr_orig)
    elif audio_path_proc is not None:
        # 从文件加载处理后音频
        y_proc, sr_proc = librosa.load(audio_path_proc, sr=16000)
        if sr_proc != sr_orig:
            y_proc = librosa.resample(y_proc, orig_sr=sr_proc, target_sr=sr_orig)
    else:
        raise ValueError("必须提供处理后的音频，通过audio_path_proc或y_proc参数")
    
    # 确保音频长度一致（取较短的）
    min_len = min(len(y_orig), len(y_proc))
    y_orig = y_orig[:min_len]
    y_proc = y_proc[:min_len]
    
    # 计算特征
    features_orig = compute_bark_features(y_orig, sr_orig)
    features_proc = compute_bark_features(y_proc, sr_orig)
    
    # 比较特征并计算损失
    losses = compare_bark_features(features_orig, features_proc)
    
    # 可视化对比
    visualize_audio_comparison(y_orig, y_proc, sr_orig, "降噪处理前后音频对比")
    
    # 打印损失报告
    print("=" * 50)
    print("狗吠特征损失分析报告")
    print("=" * 50)
    print(f"狗吠主要频率范围:     {features_orig['bark_freq_range'][0]}Hz - {features_orig['bark_freq_range'][1]}Hz")
    print("-" * 50)
    print(f"MFCC均值余弦距离:     {losses['mfcc_mean_loss']:.4f}")
    print(f"MFCC标准差余弦距离:   {losses['mfcc_std_loss']:.4f}")
    print(f"谱质心相对变化:       {losses['centroid_mean_change_pct']:.2f}%")
    print(f"零交叉率相对变化:     {losses['zcr_mean_change_pct']:.2f}%")
    print(f"谱带宽相对变化:       {losses['bandwidth_mean_change_pct']:.2f}%")
    print(f"谱滚降点相对变化:     {losses['rolloff_mean_change_pct']:.2f}%")
    print(f"RMS能量相对变化:      {losses['rms_mean_change_pct']:.2f}%")
    print("-" * 50)
    print(f"总体特征损失评分:     {losses['total_feature_loss']:.4f}")
    print("=" * 50)
    
    # 损失评估
    if losses['total_feature_loss'] < 0.1:
        print("评估结果: 损失很小，处理质量优秀")
    elif losses['total_feature_loss'] < 0.2:
        print("评估结果: 损失较小，处理质量良好")
    elif losses['total_feature_loss'] < 0.3:
        print("评估结果: 损失中等，处理质量一般")
    else:
        print("评估结果: 损失较大，可能需要调整处理参数")
    
    return losses

# 示例用法
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python comparison_visualizer.py <原始音频文件路径> [处理后音频文件路径]")
        print("示例: python comparison_visualizer.py D:\\原始音频.wav D:\\处理后音频.wav")
        sys.exit(1)
    
    audio_path_orig = sys.argv[1]
    audio_path_proc = sys.argv[2] if len(sys.argv) > 2 else None
    
    if audio_path_proc:
        # 比较两个音频文件
        analyze_and_visualize_audio_loss(audio_path_orig, audio_path_proc)
    else:
        # 处理原始音频并进行自我比较（示例）
        y_orig, sr_orig = librosa.load(audio_path_orig, sr=16000)
        # 示例处理：应用温和的带通滤波
        y_proc = bandpass(y_orig, sr_orig, low=200, high=7500, order=4)
        analyze_and_visualize_audio_loss(audio_path_orig, y_proc=y_proc, sr_proc=sr_orig)

"""
# 比较两个音频文件（原始 vs 处理后）
python detect_similar_sounds/tools/comparison_visualizer.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\outdoor_braking.mp3 D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\processed_template_preserving\outdoor_braking_clean_preserving.wav

"""