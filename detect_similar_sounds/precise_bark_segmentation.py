# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/18 上午10:00
@Author  : Kend
@FileName: precise_bark_segmentation.py
@Software: PyCharm
@modifier: Qwen3 coder
"""

"""
精确狗吠片段检测和裁剪工具 - 有待优化
用于从音频中精确检测并裁剪出狗吠片段
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
from scipy import signal


def detect_bark_onset_features(y, sr):
    """
    检测狗吠的起始特征
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
    
    Returns:
        dict: 包含各种起始特征的字典
    """
    # 计算短时能量
    frame_length = int(0.02 * sr)  # 20ms
    hop_length = int(0.01 * sr)    # 10ms
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 计算频谱质心
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                          n_fft=frame_length, 
                                                          hop_length=hop_length)[0]
    
    # 计算零交叉率
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 计算频谱通量（谱变化率）
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    
    # 确保所有特征具有相同的帧数
    min_frames = min(len(energy), len(spectral_centroids), len(zcr), len(flux))
    energy = energy[:min_frames]
    spectral_centroids = spectral_centroids[:min_frames]
    zcr = zcr[:min_frames]
    flux = flux[:min_frames]
    
    return {
        'energy': energy,
        'spectral_centroid': spectral_centroids,
        'zcr': zcr,
        'spectral_flux': flux,
        'times': librosa.frames_to_time(range(min_frames), sr=sr, hop_length=hop_length)
    }

def detect_precise_bark_segments(y, sr, 
                                energy_threshold=0.005,     # 降低阈值
                                min_duration=0.03,          # 缩短最小持续时间
                                max_duration=1.5):          # 增加最大持续时间
    """
    精确检测狗吠片段
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        energy_threshold (float): 能量阈值
        onset_threshold (float): 起始检测阈值
        min_duration (float): 最小持续时间(秒)
        max_duration (float): 最大持续时间(秒)
    
    Returns:
        list: 狗吠片段的起始和结束时间列表 [(start_time, end_time), ...]
    """
    # 提取特征
    features = detect_bark_onset_features(y, sr)
    
    # 组合特征用于检测
    # 标准化特征
    norm_energy = features['energy'] / (np.max(features['energy']) + 1e-8)
    norm_centroid = features['spectral_centroid'] / (np.max(features['spectral_centroid']) + 1e-8)
    norm_flux = features['spectral_flux'] / (np.max(features['spectral_flux']) + 1e-8)
    norm_zcr = features['zcr'] / (np.max(features['zcr']) + 1e-8)
    
    # 组合得分（狗吠通常具有高能量、高频谱质心、高谱通量）
    # 调整权重以适应处理后的音频
    combined_score = 0.35 * norm_energy + 0.3 * norm_centroid + 0.25 * norm_flux + 0.1 * (1 - norm_zcr)
    
    # 检测起始点，使用更敏感的参数
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=combined_score, 
                                       hop_length=int(0.01 * sr), units='time',
                                       backtrack=True)  # 启用回溯以获得更精确的起始点
    
    # 基于能量和组合得分进一步精确定位
    bark_segments = []
    
    for onset_time in onsets:
        # 转换为帧索引
        onset_frame = librosa.time_to_frames(onset_time, sr=sr, hop_length=int(0.01 * sr))
        
        # 确保帧索引在有效范围内
        if onset_frame >= len(combined_score):
            continue
            
        # 向前搜索结束点
        start_frame = max(0, onset_frame - 2)  # 给一点前置时间
        end_frame = onset_frame
        
        # 向后搜索直到能量下降到更低的阈值
        for i in range(onset_frame, len(combined_score)):
            if combined_score[i] < energy_threshold * 0.5:  # 使用更低的结束阈值
                break
            end_frame = i
        
        # 确保最小持续时间
        if end_frame > start_frame:
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=int(0.01 * sr))
            end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=int(0.01 * sr))
            
            # 检查持续时间是否在合理范围内
            duration = end_time - start_time
            if min_duration <= duration <= max_duration:
                # 精确调整边界
                precise_start, precise_end = refine_segment_boundaries(y, sr, start_time, end_time)
                bark_segments.append((precise_start, precise_end))
    
    # 合并重叠或相近的片段
    if len(bark_segments) > 1:
        merged_segments = [bark_segments[0]]
        for i in range(1, len(bark_segments)):
            last_seg = merged_segments[-1]
            curr_seg = bark_segments[i]
            
            # 如果片段间隔小于150ms，则合并（放宽合并条件）
            if curr_seg[0] - last_seg[1] < 0.15:
                merged_segments[-1] = (last_seg[0], curr_seg[1])
            else:
                merged_segments.append(curr_seg)
        bark_segments = merged_segments
    
    return bark_segments

def refine_segment_boundaries(y, sr, start_time, end_time):
    """
    精确调整片段边界
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        start_time (float): 初始起始时间
        end_time (float): 初始结束时间
    
    Returns:
        tuple: 精确的起始和结束时间
    """
    # 转换为样本索引
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    
    # 确保索引在有效范围内
    start_idx = max(0, start_idx)
    end_idx = min(len(y), end_idx)
    
    # 提取片段
    segment = y[start_idx:end_idx]
    
    # 计算片段的能量包络
    envelope = np.abs(signal.hilbert(segment))
    
    # 找到能量阈值（最大能量的0.5%）
    energy_threshold = np.max(envelope) * 0.005
    
    # 精确的起始点（第一个超过阈值的点）
    precise_start_offset = 0
    for i, amp in enumerate(envelope):
        if amp > energy_threshold:
            precise_start_offset = max(0, i - int(0.005 * sr))  # 给5ms的前置缓冲
            break
    
    # 精确的结束点（最后一个超过阈值的点）
    precise_end_offset = len(envelope)
    for i in range(len(envelope) - 1, -1, -1):
        if envelope[i] > energy_threshold:
            precise_end_offset = min(len(envelope), i + int(0.015 * sr))  # 给15ms的后置缓冲
            break
    
    # 转换回时间
    precise_start = start_time + precise_start_offset / sr
    precise_end = start_time + precise_end_offset / sr
    
    return precise_start, precise_end

def visualize_bark_segments(y, sr, segments, title="狗吠片段检测结果"):
    """
    可视化狗吠片段检测结果
    
    Args:
        y (np.array): 音频信号
        sr (int): 采样率
        segments (list): 狗吠片段列表 [(start_time, end_time), ...]
        title (str): 图形标题
    """
    plt.figure(figsize=(15, 8))
    
    # 显示波形
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    
    # 标记检测到的片段
    for i, (start, end) in enumerate(segments):
        plt.axvspan(start, end, alpha=0.3, color='red', label=f'片段 {i+1}' if i == 0 else "")
    
    plt.title(title)
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.legend()
    plt.tight_layout()
    plt.show()

def extract_and_save_bark_segments(audio_path, output_dir=None, visualize=True):
    """
    提取并保存狗吠片段
    
    Args:
        audio_path (str): 音频文件路径
        output_dir (str): 输出目录，如果为None则在原文件同目录下创建子目录
        visualize (bool): 是否可视化结果
    
    Returns:
        list: 保存的片段文件路径列表
    """
    # 加载音频
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 检测狗吠片段
    segments = detect_precise_bark_segments(y, sr)
    
    if len(segments) == 0:
        print("未检测到任何狗吠片段")
        return []
    
    print(f"检测到 {len(segments)} 个狗吠片段:")
    for i, (start, end) in enumerate(segments):
        print(f"  片段 {i+1}: {start:.3f}s - {end:.3f}s (持续时间: {end-start:.3f}s)")
    
    # 确定输出目录
    if output_dir is None:
        base_dir = os.path.dirname(audio_path)
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(base_dir, f"{filename}_bark_segments")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存片段
    saved_files = []
    for i, (start, end) in enumerate(segments):
        # 转换为样本索引
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        # 提取片段
        segment = y[start_idx:end_idx]
        
        # 保存片段
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        segment_filename = f"{base_name}_bark_{i+1:02d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        
        sf.write(segment_path, segment, sr)
        saved_files.append(segment_path)
        print(f"已保存片段: {segment_path}")
    
    # 可视化结果
    if visualize:
        visualize_bark_segments(y, sr, segments, "狗吠片段检测结果")
    
    return saved_files

# 示例用法
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python precise_bark_segmentation.py <音频文件路径> [输出目录]")
        print("示例: python precise_bark_segmentation.py D:\\音频.wav D:\\输出目录")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_and_save_bark_segments(audio_path, output_dir)


"""

python detect_similar_sounds/precise_bark_segmentation.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\processed_template_preserving\outdoor_braking_clean_preserving.wav

"""