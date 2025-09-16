#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级狗吠声纹识别算法 - 优化版
基于物理和数学方法，不依赖深度学习，专为MCU部署优化

改进点：
1. 多段embedding平均池化
2. MFCC+Δ+ΔΔ + log-Mel特征拼接
3. 增加短时能量/过零率特征
4. 优化的GMM-UBM模型+MAP adaptation
5. score normalization (z-norm)
6. 改进的阈值选择策略
"""

import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis, norm
import pickle
import time
from collections import defaultdict

# ================== 配置 ==================
class Config:
    def __init__(self):
        # 音频参数
        self.SR = 16000  # 采样率
        self.N_MFCC = 13  # MFCC特征数量
        self.N_MELS = 26  # Mel滤波器数量 - 优化配置避免空滤波器
        self.FRAME_LEN = 0.025  # 帧长(秒)
        self.FRAME_STEP = 0.01  # 帧移(秒)
        self.FMIN = 300.0  # 带通下限(Hz) - 适合狗吠频率范围
        self.FMAX = 8000.0  # 带通上限(Hz)
        
        # 模型参数
        self.UBM_COMPONENTS = 16  # UBM高斯分量数
        self.DOG_GMM_COMPONENTS = 5  # 每只狗的GMM分量数
        self.REG_COVAR = 1e-3  # 协方差正则化
        self.MAP_WEIGHT = 0.5  # MAP适应权重 - 更倾向于UBM的先验知识
        
        # VAD和预处理参数
        self.VAD_RMS_THRESHOLD = 0.002  # 语音活动检测阈值
        self.TARGET_RMS = 0.1  # 音量归一化目标值
        self.TRIM_TOP_DB = 30  # 静音修剪阈值
        
        # 特征优化参数
        self.USE_ENERGY_NORMALIZATION = True  # 是否使用能量归一化
        self.USE_CEPSTRAL_NORMALIZATION = True  # 是否使用倒谱归一化
        self.USE_DYNAMIC_RANGE_COMPRESSION = True  # 是否使用动态范围压缩
        
        # 特征池化参数
        self.SEGMENT_DURATION = 0.5  # 分段池化的段长(秒)
        self.OVERLAP_RATIO = 0.3  # 重叠比例
        
        # 阈值策略参数
        self.BACKGROUND_THRESH = -85  # 背景判定阈值
        self.MIN_VOICE_DURATION = 0.2  # 最小语音持续时间(秒)
        
        # 后处理参数
        self.USE_Z_SCORE_NORM = True  # 是否使用z-score归一化
        self.USE_LDA = False  # 是否使用LDA降维（如需更轻量可关闭）
        self.LDA_N_COMPONENTS = None  # LDA组件数量

# 全局配置实例
cfg = Config()

# ================== 音频预处理 ==================

def butter_bandpass(lowcut, highcut, fs, order=4):
    """设计巴特沃斯带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=cfg.FMIN, highcut=cfg.FMAX, fs=cfg.SR):
    """应用带通滤波器，保留狗吠的主要频率范围"""
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=4)
        y = filtfilt(b, a, data)  # 零相位滤波，避免相位失真
        return y
    except Exception:
        return data


def normalize_rms(y, target_rms=cfg.TARGET_RMS):
    """RMS能量归一化"""
    # 确保y是numpy数组
    if not isinstance(y, np.ndarray):
        try:
            y = np.array(y, dtype=np.float32)
        except Exception:
            return y  # 如果转换失败，返回原始数据
    
    rms = np.sqrt(np.mean(y ** 2) + 1e-9)
    if rms < 1e-9:
        return y
    return y * (target_rms / (rms + 1e-9))


def dynamic_range_compression(y, threshold=0.02, ratio=2.0):
    """动态范围压缩，增强弱信号同时限制强信号"""
    y_compressed = np.copy(y)
    # 对超过阈值的部分进行压缩
    y_compressed[y > threshold] = threshold + (y[y > threshold] - threshold) / ratio
    y_compressed[y < -threshold] = -threshold + (y[y < -threshold] + threshold) / ratio
    return y_compressed


def simple_vad(y, frame_length=1024, hop_length=512, threshold=cfg.VAD_RMS_THRESHOLD):
    """增强版语音活动检测，更好地检测可能包含狗叫的声音片段"""
    # 确保帧长适合采样率
    frame_length = min(frame_length, len(y))
    hop_length = min(hop_length, frame_length // 2)
    
    # 计算每帧的RMS能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 如果没有帧，返回空列表
    if len(rms) == 0:
        return []
    
    # 计算动态阈值，更鲁棒的方法
    peak_rms = np.max(rms)
    if peak_rms < threshold * 2:
        return []
    
    # 自适应阈值：结合全局和局部信息
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    dynamic_threshold = max(threshold, min(mean_rms + 2.0 * std_rms, peak_rms * 0.3))
    
    # 检测语音帧
    voice_frames = rms > dynamic_threshold
    
    # 应用中值滤波平滑，去除孤立的噪声帧
    from scipy.ndimage import median_filter
    voice_frames = median_filter(voice_frames, size=3)
    
    # 找到连续的语音段
    voice_segments = []
    current_start = None
    
    # 计算每个语音帧的实际时间位置
    frame_times = librosa.frames_to_time(np.arange(len(voice_frames)), sr=cfg.SR, hop_length=hop_length)
    
    for i, is_voice in enumerate(voice_frames):
        if is_voice and current_start is None:
            current_start = i
        elif not is_voice and current_start is not None:
            current_end = i
            # 计算语音段时长
            if current_end > current_start:
                segment_duration = frame_times[current_end-1] - frame_times[current_start]
                
                # 根据语音段长度调整阈值要求
                adjusted_min_duration = cfg.MIN_VOICE_DURATION * 0.8 if segment_duration < 0.5 else cfg.MIN_VOICE_DURATION
                
                if segment_duration >= adjusted_min_duration:
                    # 转换为采样点索引
                    start_sample = int(frame_times[current_start] * cfg.SR)
                    end_sample = min(int(frame_times[current_end-1] * cfg.SR) + frame_length, len(y))
                    voice_segments.append((start_sample, end_sample))
            current_start = None
    
    # 处理最后一个段
    if current_start is not None and current_start < len(voice_frames):
        current_end = len(voice_frames)
        segment_duration = frame_times[-1] - frame_times[current_start]
        
        adjusted_min_duration = cfg.MIN_VOICE_DURATION * 0.8 if segment_duration < 0.5 else cfg.MIN_VOICE_DURATION
        
        if segment_duration >= adjusted_min_duration:
            start_sample = int(frame_times[current_start] * cfg.SR)
            end_sample = len(y)
            voice_segments.append((start_sample, end_sample))
    
    # 合并相邻的短语音段
    if len(voice_segments) > 1:
        merged_segments = [voice_segments[0]]
        for curr_start, curr_end in voice_segments[1:]:
            prev_start, prev_end = merged_segments[-1]
            # 如果两个段之间的间隔小于0.1秒，合并它们
            if curr_start - prev_end < 0.1 * cfg.SR:
                merged_segments[-1] = (prev_start, curr_end)
            else:
                merged_segments.append((curr_start, curr_end))
        return merged_segments
    
    return voice_segments


def preprocess_audio(y):
    """完整的音频预处理流程"""
    # 确保y是numpy数组
    if not isinstance(y, np.ndarray):
        try:
            y = np.array(y, dtype=np.float32)
        except Exception:
            return y  # 如果转换失败，返回原始数据
    
    # 1. 静音修剪
    if cfg.TRIM_TOP_DB is not None and len(y) > 0:
        try:
            y, _ = librosa.effects.trim(y, top_db=cfg.TRIM_TOP_DB)
        except Exception:
            pass  # 如果修剪失败，继续处理原始音频
    
    # 2. 带通滤波
    y = bandpass_filter(y)
    
    # 3. 动态范围压缩（如果启用）
    if cfg.USE_DYNAMIC_RANGE_COMPRESSION:
        y = dynamic_range_compression(y)
    
    # 4. 语音活动检测
    voice_segments = simple_vad(y)
    
    # 如果没有检测到语音段，返回原始音频的一小部分
    if not voice_segments:
        return y[:1000] if len(y) > 1000 else y
    
    # 合并所有语音段
    merged_audio = []
    for start, end in voice_segments:
        merged_audio.extend(y[start:end])
    
    # 如果合并后的音频为空，返回原始音频的一小部分
    if not merged_audio:
        return y[:1000] if len(y) > 1000 else y
    
    y = np.array(merged_audio)
    
    # 5. RMS能量归一化
    if cfg.USE_ENERGY_NORMALIZATION:
        y = normalize_rms(y)
    
    return y

# ================== 增强特征提取 ==================

def extract_enhanced_features(y):
    """提取增强的特征集：MFCC+Δ+ΔΔ + log-Mel + 能量/过零率特征"""
    try:
        # 确保n_fft为2的幂次，以优化计算效率
        n_fft = int(2 ** np.ceil(np.log2(cfg.FRAME_LEN * cfg.SR)))
        hop_length = int(cfg.SR * cfg.FRAME_STEP)
        
        # 1. 提取MFCC特征
        mfcc_feat = librosa.feature.mfcc(
            y=y,
            sr=cfg.SR,
            n_mfcc=cfg.N_MFCC,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=min(cfg.FMAX, cfg.SR/2),  # 确保不超过奈奎斯特频率
            htk=True
        ).T  # 转置为 (帧数, 特征数)
        
        # 2. 提取log-Mel谱特征
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=cfg.SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=min(cfg.FMAX, cfg.SR/2)
        )
        log_mel = librosa.power_to_db(mel_spectrogram).T  # 转置为 (帧数, 特征数)
        
        # 3. 计算一阶和二阶差分特征
        if mfcc_feat.shape[0] >= 9:
            # 正常情况，使用默认宽度9
            mfcc_delta = librosa.feature.delta(mfcc_feat.T).T
            mfcc_delta2 = librosa.feature.delta(mfcc_feat.T, order=2).T
            logmel_delta = librosa.feature.delta(log_mel.T).T
            logmel_delta2 = librosa.feature.delta(log_mel.T, order=2).T
        else:
            # 帧数较少时，使用较小的窗口宽度
            width = min(5, mfcc_feat.shape[0] - 1)  # 确保width是奇数且不超过帧数
            if width < 3:
                width = 3  # 最小窗口宽度为3
            
            try:
                mfcc_delta = librosa.feature.delta(mfcc_feat.T, width=width).T
                mfcc_delta2 = librosa.feature.delta(mfcc_feat.T, order=2, width=width).T
                logmel_delta = librosa.feature.delta(log_mel.T, width=width).T
                logmel_delta2 = librosa.feature.delta(log_mel.T, order=2, width=width).T
            except Exception:
                # 如果计算失败，使用零矩阵作为替代
                mfcc_delta = np.zeros_like(mfcc_feat)
                mfcc_delta2 = np.zeros_like(mfcc_feat)
                logmel_delta = np.zeros_like(log_mel)
                logmel_delta2 = np.zeros_like(log_mel)
        
        # 4. 提取短时能量特征
        energy = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        energy = energy.reshape(-1, 1)  # 转换为 (帧数, 1)
        
        # 5. 提取过零率特征
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        zcr = zcr.reshape(-1, 1)  # 转换为 (帧数, 1)
        
        # 确保所有特征的帧数一致
        min_frames = min(mfcc_feat.shape[0], log_mel.shape[0], energy.shape[0], zcr.shape[0])
        
        # 组合所有特征
        features = np.concatenate([
            mfcc_feat[:min_frames], 
            mfcc_delta[:min_frames], 
            mfcc_delta2[:min_frames],
            log_mel[:min_frames],
            logmel_delta[:min_frames],
            logmel_delta2[:min_frames],
            energy[:min_frames],
            zcr[:min_frames]
        ], axis=1)
        
        # 6. 倒谱均值和方差归一化 (CMVN)
        if cfg.USE_CEPSTRAL_NORMALIZATION:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-9
            features = (features - mean) / std
        
        return features
    except Exception as e:
        print(f"[ERROR] 特征提取过程中出错: {str(e)}")
        # 返回一个最小有效特征矩阵作为回退
        return np.zeros((5, cfg.N_MFCC + cfg.N_MFCC * 2 + cfg.N_MELS * 3 + 2))  # MFCC+Δ+ΔΔ + log-Mel+Δ+ΔΔ + 能量+过零率


def extract_pitch_features(y):
    """提取基频特征统计量"""
    try:
        # 使用PYIN算法提取基频
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=5000, sr=cfg.SR)
        
        # 过滤掉未检测到基频的帧
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # 计算基频统计量
        f0_mean = np.mean(f0_valid)
        f0_std = np.std(f0_valid)
        f0_min = np.min(f0_valid)
        f0_max = np.max(f0_valid)
        
        return np.array([f0_mean, f0_std, f0_min, f0_max])
    except Exception:
        return np.array([0.0, 0.0, 0.0, 0.0])


def extract_acoustic_stats(features):
    """计算特征的统计量"""
    if features is None or len(features) == 0:
        return np.zeros(features.shape[1] * 4) if features is not None else np.array([])
    
    # 计算统计特征
    mean_ = np.mean(features, axis=0)
    std_ = np.std(features, axis=0)
    
    # 确保有足够的数据点计算偏度和峰度
    if len(features) > 3:
        skew_ = skew(features, axis=0)
        kurt_ = kurtosis(features, axis=0)
    else:
        skew_ = np.zeros_like(mean_)
        kurt_ = np.zeros_like(mean_)
    
    # 处理可能的NaN值
    mean_ = np.nan_to_num(mean_, nan=0.0)
    std_ = np.nan_to_num(std_, nan=0.0)
    skew_ = np.nan_to_num(skew_, nan=0.0)
    kurt_ = np.nan_to_num(kurt_, nan=0.0)
    
    # 组合所有统计特征
    stats = np.concatenate([mean_, std_, skew_, kurt_], axis=0)
    
    return stats


def generate_segment_embeddings(y):
    """生成多段音频的嵌入向量，用于后续平均池化"""
    embeddings = []
    
    # 分段处理
    segment_samples = int(cfg.SEGMENT_DURATION * cfg.SR)
    overlap_samples = int(segment_samples * cfg.OVERLAP_RATIO)
    step_samples = segment_samples - overlap_samples
    
    # 如果音频太短，直接返回单个嵌入
    if len(y) < segment_samples * 1.5:
        features = extract_enhanced_features(y)
        acoustic_stats = extract_acoustic_stats(features)
        pitch_feats = extract_pitch_features(y)
        
        # 组合所有特征
        embedding = np.concatenate([acoustic_stats, pitch_feats], axis=0)
        
        # L2归一化
        norm = np.linalg.norm(embedding) + 1e-9
        if norm > 0:
            embedding = embedding / norm
        
        embeddings.append(embedding)
    else:
        # 多段处理
        for i in range(0, len(y) - segment_samples + 1, step_samples):
            segment = y[i:i+segment_samples]
            
            features = extract_enhanced_features(segment)
            acoustic_stats = extract_acoustic_stats(features)
            pitch_feats = extract_pitch_features(segment)
            
            # 组合所有特征
            embedding = np.concatenate([acoustic_stats, pitch_feats], axis=0)
            
            # L2归一化
        norm = np.linalg.norm(embedding) + 1e-9
        if norm > 0:
            embedding = embedding / norm
        
        embeddings.append(embedding)
    
    return embeddings

# ================== 高级GMM-UBM模型 ==================

class AdvancedDogVoiceModel:
    def __init__(self):
        self.ubm = None
        self.dog_gmms = {}
        self.scaler = StandardScaler()
        self.lda = None if not cfg.USE_LDA else LinearDiscriminantAnalysis(n_components=cfg.LDA_N_COMPONENTS)
        self.trained = False
        # 用于z-norm的统计信息
        self.score_stats = defaultdict(lambda: {'mean': 0.0, 'std': 1.0})
        self.enroll_data = {}
        
    def train_ubm(self, all_features):
        """训练UBM模型"""
        print(f"[INFO] 训练UBM模型，使用 {all_features.shape[0]} 帧数据...")
        
        # 特征标准化
        self.scaler.fit(all_features)
        scaled_features = self.scaler.transform(all_features)
        
        # 训练UBM
        self.ubm = GaussianMixture(
            n_components=cfg.UBM_COMPONENTS,
            covariance_type='diag',
            max_iter=200,
            reg_covar=cfg.REG_COVAR,
            random_state=42
        )
        
        self.ubm.fit(scaled_features)
        print("[INFO] UBM模型训练完成")
        
    def train_lda(self, features_list, labels_list):
        """训练LDA模型用于特征降维和判别"""
        if not cfg.USE_LDA or self.lda is None:
            return
        
        print("[INFO] 训练LDA模型...")
        try:
            # 确保labels是整数编码
            unique_labels = list(set(labels_list))
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            encoded_labels = [label_to_idx[label] for label in labels_list]
            
            # 特征标准化
            all_features = np.vstack(features_list)
            scaled_features = self.scaler.transform(all_features)
            
            # 训练LDA
            self.lda.fit(scaled_features, encoded_labels)
            print("[INFO] LDA模型训练完成")
        except Exception as e:
            print(f"[ERROR] LDA训练失败: {str(e)}")
            self.lda = None
            cfg.USE_LDA = False
            
    def apply_lda(self, features):
        """应用LDA变换"""
        if not cfg.USE_LDA or self.lda is None:
            return features
        
        try:
            return self.lda.transform(features)
        except Exception:
            return features
        
    def train_dog_models(self, enroll_data):
        """为每只狗训练GMM模型，使用增强的MAP适应"""
        if self.ubm is None:
            raise ValueError("请先训练UBM模型")
        
        # 保存注册数据用于后续score normalization
        self.enroll_data = enroll_data
        
        # 为LDA准备数据
        lda_features = []
        lda_labels = []
        
        for dog_id, features in enroll_data.items():
            print(f"[INFO] 为狗 {dog_id} 训练GMM模型...")
            
            # 特征标准化
            scaled_features = self.scaler.transform(features)
            
            # 使用UBM参数初始化，但要对权重进行归一化
            ubm_weights = self.ubm.weights_[:cfg.DOG_GMM_COMPONENTS]
            normalized_weights = ubm_weights / np.sum(ubm_weights)
            
            dog_gmm = GaussianMixture(
                n_components=cfg.DOG_GMM_COMPONENTS,
                covariance_type='diag',
                max_iter=100,
                reg_covar=cfg.REG_COVAR,
                random_state=42,
                means_init=self.ubm.means_[:cfg.DOG_GMM_COMPONENTS],
                precisions_init=self.ubm.precisions_[:cfg.DOG_GMM_COMPONENTS],
                weights_init=normalized_weights
            )
            
            # 训练狗特定的GMM
            dog_gmm.fit(scaled_features)
            
            # 应用增强的MAP适应 - 不仅仅更新均值，还考虑方差
            if cfg.MAP_WEIGHT < 1.0:
                # 对均值进行MAP适应
                dog_gmm.means_ = cfg.MAP_WEIGHT * dog_gmm.means_ + (1 - cfg.MAP_WEIGHT) * self.ubm.means_[:cfg.DOG_GMM_COMPONENTS]
                
                # 对协方差也进行轻度的MAP适应
                ubm_covars = np.sqrt(1.0 / self.ubm.precisions_[:cfg.DOG_GMM_COMPONENTS])
                dog_covars = np.sqrt(1.0 / dog_gmm.precisions_)
                adapted_covars = cfg.MAP_WEIGHT * dog_covars + (1 - cfg.MAP_WEIGHT) * ubm_covars
                dog_gmm.precisions_ = 1.0 / (adapted_covars ** 2)
            
            self.dog_gmms[dog_id] = dog_gmm
            
            # 为LDA添加数据
            lda_features.append(scaled_features)
            lda_labels.extend([dog_id] * scaled_features.shape[0])
            
            print(f"[INFO] 狗 {dog_id} 的模型训练完成")
        
        # 训练LDA
        self.train_lda(lda_features, lda_labels)
        
        # 计算score normalization的统计信息
        self._compute_score_stats()
        
        self.trained = True
        
    def _compute_score_stats(self):
        """计算score normalization的统计信息"""
        if not self.trained or not self.dog_gmms:
            return
        
        print("[INFO] 计算score normalization统计信息...")
        
        # 为每只狗计算genuine和impostor分数
        for target_id, target_gmm in self.dog_gmms.items():
            genuine_scores = []
            impostor_scores = []
            
            # 收集genuine分数
            if target_id in self.enroll_data:
                target_features = self.scaler.transform(self.enroll_data[target_id])
                genuine_scores.extend(target_gmm.score_samples(target_features))
            
            # 收集impostor分数
            for impostor_id, impostor_features in self.enroll_data.items():
                if impostor_id != target_id:
                    impostor_scaled = self.scaler.transform(impostor_features)
                    impostor_scores.extend(target_gmm.score_samples(impostor_scaled))
            
            # 计算统计信息
            if genuine_scores and impostor_scores:
                all_scores = genuine_scores + impostor_scores
                self.score_stats[target_id]['mean'] = np.mean(all_scores)
                self.score_stats[target_id]['std'] = np.std(all_scores) + 1e-9
        
        print("[INFO] score normalization统计信息计算完成")
        
    def enroll_dog(self, dog_id, audio_files):
        """注册新的狗，支持多场景录音"""
        features = []
        
        # 数据增强，模拟不同场景
        for file_path in audio_files:
            # 加载和预处理音频
            y, _ = librosa.load(file_path, sr=cfg.SR)
            y = preprocess_audio(y)
            
            # 提取特征
            enhanced_feats = extract_enhanced_features(y)
            features.append(enhanced_feats)
            
            # 简单数据增强 - 轻微改变音频特性
            for factor in [0.9, 1.1]:  # 音量扰动
                y_aug = y * factor
                aug_feats = extract_enhanced_features(y_aug)
                features.append(aug_feats)
        
        if not features:
            return False
        
        # 合并所有特征
        all_feats = np.vstack(features)
        
        # 如果还没有UBM，先创建简单的UBM
        if self.ubm is None:
            self.scaler.fit(all_feats)
            scaled_feats = self.scaler.transform(all_feats)
            
            self.ubm = GaussianMixture(
                n_components=min(8, len(scaled_feats)//10),  # 简单UBM
                covariance_type='diag',
                max_iter=100,
                reg_covar=cfg.REG_COVAR,
                random_state=42
            )
            self.ubm.fit(scaled_feats)
        
        # 训练这只狗的模型
        self.train_dog_models({dog_id: all_feats})
        return True
        
    def recognize(self, y):
        """高级识别函数：多段embedding池化 + 优化评分 - 修改版"""
        try:
            if not self.trained or not self.dog_gmms:
                return None, None
            
            # 输入验证
            if y is None or len(y) == 0:
                return 'background', {}
            
            # 确保y是numpy数组
            if not isinstance(y, np.ndarray):
                try:
                    y = np.array(y, dtype=np.float32)
                except Exception:
                    return 'background', {}  # 如果转换失败，返回背景
            
            # 预处理音频 - 与dog_separability_analysis.py保持一致的预处理
            # 1. 预加重
            y = np.append(y[0], y[1:] - 0.97*y[:-1])
            # 2. 静音修剪
            y, _ = librosa.effects.trim(y, top_db=30)
            
            # 计算音频实际时长
            audio_duration = librosa.get_duration(y=y, sr=cfg.SR)
            
            # 使用与dog_separability_analysis.py相同的特征提取方法
            # 提取MFCC和log-Mel特征，并计算统计量
            try:
                # 确保n_fft为2的幂次
                n_fft = int(2 ** np.ceil(np.log2(cfg.FRAME_LEN * cfg.SR)))
                hop_length = int(cfg.SR * cfg.FRAME_STEP)
                
                # MFCC特征
                mfcc = librosa.feature.mfcc(
                    y=y, sr=cfg.SR, n_mfcc=20, n_fft=n_fft, hop_length=hop_length,
                    n_mels=40, fmin=cfg.FMIN, fmax=min(cfg.FMAX, cfg.SR/2)
                )
                
                # log-Mel特征
                mel = librosa.feature.melspectrogram(
                    y=y, sr=cfg.SR, n_fft=n_fft, hop_length=hop_length,
                    n_mels=40, fmin=cfg.FMIN, fmax=min(cfg.FMAX, cfg.SR/2)
                )
                logmel = librosa.power_to_db(mel)
                
                # 转置并对齐时间长度
                mfcc_t = mfcc.T
                logmel_t = logmel.T
                T = min(mfcc_t.shape[0], logmel_t.shape[0])
                if T <= 0:
                    return 'background', {}
                
                mfcc_t = mfcc_t[:T]
                logmel_t = logmel_t[:T]
                
                # 拼接特征
                feat = np.concatenate([mfcc_t, logmel_t], axis=1)
                
                # 计算统计量 - 与dog_separability_analysis.py保持一致
                mu = np.mean(feat, axis=0)
                sd = np.std(feat, axis=0)
                sk = skew(feat, axis=0)
                kt = kurtosis(feat, axis=0)
                
                # 组合成嵌入向量
                embedding = np.concatenate([mu, sd, sk, kt])
                
                # L2归一化
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # 标准化
                scaler = StandardScaler()
                embedding_2d = embedding.reshape(1, -1)
                embedding_scaled = scaler.fit_transform(embedding_2d)[0]
                
            except Exception as e:
                print(f"[ERROR] 特征提取失败: {str(e)}")
                return 'background', {}
            
            # 计算每只狗模板与当前嵌入的余弦相似度 - 与dog_separability_analysis.py保持一致
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = {}
            
            # 为每只狗创建模板（平均嵌入）
            for dog_id in self.dog_gmms.keys():
                if dog_id in self.enroll_data:
                    # 获取这只狗的所有注册特征
                    dog_features = self.enroll_data[dog_id]
                    
                    # 为注册特征计算统计量，与测试特征保持一致的处理
                    dog_mu = np.mean(dog_features, axis=0)
                    dog_sd = np.std(dog_features, axis=0)
                    dog_sk = skew(dog_features, axis=0)
                    dog_kt = kurtosis(dog_features, axis=0)
                    
                    # 组合成狗的模板嵌入
                    dog_embedding = np.concatenate([dog_mu, dog_sd, dog_sk, dog_kt])
                    
                    # L2归一化
                    dog_norm = np.linalg.norm(dog_embedding)
                    if dog_norm > 0:
                        dog_embedding = dog_embedding / dog_norm
                    
                    # 标准化
                    dog_embedding_2d = dog_embedding.reshape(1, -1)
                    dog_embedding_scaled = scaler.transform(dog_embedding_2d)[0]
                    
                    # 计算余弦相似度
                    similarity = cosine_similarity(
                        embedding_scaled.reshape(1, -1), 
                        dog_embedding_scaled.reshape(1, -1)
                    )[0][0]
                    
                    similarities[dog_id] = round(similarity, 4)
            
            if not similarities:
                return 'background', {}
            
            # 找到相似度最高的狗
            best_dog = max(similarities, key=similarities.get)
            best_score = similarities[best_dog]
            
            # 计算第二高相似度，用于判断置信度
            sorted_scores = sorted(similarities.values(), reverse=True)
            if len(sorted_scores) > 1:
                score_diff = best_score - sorted_scores[1]
            else:
                score_diff = float('inf')
            
            # 优化的阈值策略 - 基于dog_separability_analysis.py的分析结果
            # 由于AUC为1.0，genuine和impostor分布无重叠，可以使用较低的阈值
            background_threshold = 0.3  # 大幅降低背景阈值
            min_confidence_diff = 0.1   # 大幅降低置信度差异要求
            min_normalized_score = 0.4  # 归一化得分阈值，根据实际分布设置
            
            # 分类决策 - 更平衡的决策逻辑，更倾向于识别狗而不是背景
            if best_score < background_threshold:
                return 'background', similarities
            elif (best_score < min_normalized_score or score_diff < min_confidence_diff):
                # 归一化得分较低或置信度不足
                return 'possible_dog', similarities
            else:
                # 得分足够高且置信度足够高，返回识别结果
                return best_dog, similarities
        except Exception as e:
            print(f"[ERROR] 识别过程中出错: {str(e)}")
            return 'background', {}
        
    def save_model(self, path):
        """保存模型到文件"""
        model_data = {
            'ubm': self.ubm,
            'dog_gmms': self.dog_gmms,
            'scaler': self.scaler,
            'lda': self.lda if cfg.USE_LDA else None,
            'trained': self.trained,
            'score_stats': dict(self.score_stats),  # 转换为普通字典
            'enroll_data': self.enroll_data
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, path):
        """从文件加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.ubm = model_data['ubm']
        self.dog_gmms = model_data['dog_gmms']
        self.scaler = model_data['scaler']
        self.lda = model_data['lda'] if cfg.USE_LDA else None
        self.trained = model_data['trained']
        
        # 恢复score_stats和enroll_data
        if 'score_stats' in model_data:
            self.score_stats = defaultdict(lambda: {'mean': 0.0, 'std': 1.0}, model_data['score_stats'])
        if 'enroll_data' in model_data:
            self.enroll_data = model_data['enroll_data']

# ================== 工具函数 ==================

def compute_optimal_thresholds(validation_data):
    """根据验证数据计算最佳阈值"""
    from sklearn.metrics import roc_curve, auc
    
    # 准备ROC分析数据
    y_true = []
    y_scores = []
    
    for sample in validation_data:
        true_label = sample['true_label']
        predicted_label = sample['predicted_label']
        score = sample['score']
        
        # 1表示正确识别，0表示错误识别
        y_true.append(1 if true_label == predicted_label else 0)
        y_scores.append(score)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 寻找最佳阈值（Youden's J statistic）
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"[INFO] 最佳阈值: {best_threshold}, 对应的TPR: {tpr[best_idx]:.3f}, FPR: {fpr[best_idx]:.3f}, AUC: {roc_auc:.3f}")
    
    return best_threshold

# ================== 主函数 ==================

if __name__ == "__main__":
    # 示例用法
    print("高级狗吠声纹识别系统")
    
    # 创建模型实例
    model = AdvancedDogVoiceModel()
    
    # 这里可以添加具体的训练和测试代码
    # 例如：
    # enroll_dir = "./dog_voice_samples"
    # test_dir = "./test_samples"
    # 
    # # 训练模型
    # enroll_data = {}
    # all_features = []
    # # ... 加载数据并提取特征 ...
    # 
    # # 训练UBM
    # model.train_ubm(np.vstack(all_features))
    # 
    # # 训练每只狗的模型
    # model.train_dog_models(enroll_data)
    # 
    # # 保存模型
    # model.save_model("advanced_dog_voice_model.pkl")
    # 
    # # 识别测试
    # y, _ = librosa.load("test_dog_voice.wav", sr=cfg.SR)
    # result, scores = model.recognize(y)
    # print(f"识别结果: {result}")
    # print(f"得分: {scores}")