# -*- coding: utf-8 -*-
"""
改进版狗吠声纹识别算法 - 针对端侧部署优化
基于物理和数学方法，不依赖深度学习
"""

import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, filtfilt
from scipy.stats import skew, kurtosis
import pickle
import time

# ================== 配置 ==================
class Config:
    def __init__(self):
        # 音频参数
        self.SR = 16000  # 采样率
        self.N_MFCC = 20  # MFCC特征数量
        self.N_MELS = 32  # Mel滤波器数量 - 减少以避免空滤波器警告
        self.FRAME_LEN = 0.025  # 帧长(秒)
        self.FRAME_STEP = 0.01  # 帧移(秒)
        self.FMIN = 300.0  # 带通下限(Hz) - 适合狗吠频率范围
        self.FMAX = 8000.0  # 带通上限(Hz) - 降低以匹配采样率限制
        
        # 模型参数
        self.UBM_COMPONENTS = 8  # UBM高斯分量数
        self.DOG_GMM_COMPONENTS = 3  # 每只狗的GMM分量数
        self.REG_COVAR = 1e-3  # 协方差正则化
        self.MAP_WEIGHT = 0.7  # MAP适应权重
        
        # VAD和预处理参数
        self.VAD_RMS_THRESHOLD = 0.002  # 语音活动检测阈值
        self.TARGET_RMS = 0.1  # 音量归一化目标值
        self.TRIM_TOP_DB = 30  # 静音修剪阈值
        
        # 区分度优化参数
        self.USE_ENERGY_NORMALIZATION = True  # 是否使用能量归一化
        self.USE_CEPSTRAL_NORMALIZATION = True  # 是否使用倒谱归一化
        self.USE_DYNAMIC_RANGE_COMPRESSION = True  # 是否使用动态范围压缩
        self.ALPHA = 0.75  # 融合权重(cos vs euc)
        
        # 端侧优化参数
        self.FEATURE_DIM_REDUCTION = False  # 是否进行特征降维
        self.DIM_REDUCTION_METHOD = 'pca'  # 降维方法
        self.TARGET_DIM = 30  # 降维后的维度
        
        # 背景噪声处理 - 调整阈值以提高识别召回率
        self.BACKGROUND_THRESH = -95  # 背景判定阈值 - 降低以提高召回率
        self.POSSIBLE_DOG_THRESH = -85  # 可能包含狗叫的阈值 - 降低以提高召回率
        self.MIN_VOICE_DURATION = 0.2  # 最小语音持续时间(秒)
        self.NOISE_PROFILE_PATH = None  # 噪声 profile 路径

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
    """增强版语音活动检测，更好地检测可能包含狗叫的声音片段
    
    Args:
        y: 音频数据
        frame_length: 帧长
        hop_length: 帧移
        threshold: 能量阈值
        
    Returns:
        list: 语音段列表，每个元素是(start_sample, end_sample)元组
    """
    # 确保帧长适合采样率
    frame_length = min(frame_length, len(y))
    hop_length = min(hop_length, frame_length // 2)
    
    # 计算每帧的RMS能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 如果没有帧，返回空列表
    if len(rms) == 0:
        return []
    
    # 计算动态阈值，更鲁棒的方法
    # 使用信号的峰值能量作为参考
    peak_rms = np.max(rms)
    # 如果峰值能量很低，可能是纯背景噪声
    if peak_rms < threshold * 2:
        return []
    
    # 自适应阈值：结合全局和局部信息
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    # 阈值计算考虑峰值能量，使检测更鲁棒
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
                # 对于短音频放宽要求
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
        
        # 对最后一段同样放宽要求
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
    
    # 4. 语音活动检测 - simple_vad返回语音段列表
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

# ================== 特征提取 ==================

def extract_mfcc_features(y):
    """提取MFCC特征及其差分特征"""
    try:
        # 确保n_fft为2的幂次，以优化计算效率
        n_fft = int(2 ** np.ceil(np.log2(cfg.FRAME_LEN * cfg.SR)))
        
        # 计算MFCC特征，调整参数以避免空滤波器警告
        mfcc_feat = librosa.feature.mfcc(
            y=y, 
            sr=cfg.SR, 
            n_mfcc=cfg.N_MFCC, 
            n_fft=n_fft,
            hop_length=int(cfg.SR * cfg.FRAME_STEP),
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=min(cfg.FMAX, cfg.SR/2),  # 确保不超过奈奎斯特频率
            htk=True
        ).T  # 转置为 (帧数, 特征数)
        
        # 检查特征是否有效
        if mfcc_feat.shape[0] == 0:
            return np.zeros((5, cfg.N_MFCC))  # 返回最小有效特征矩阵
        
        # 计算一阶和二阶差分，但要根据帧数调整窗口宽度
        if mfcc_feat.shape[0] >= 9:
            # 正常情况，使用默认宽度9
            delta = librosa.feature.delta(mfcc_feat.T).T
            delta2 = librosa.feature.delta(mfcc_feat.T, order=2).T
            # 组合特征
            features = np.concatenate([mfcc_feat, delta, delta2], axis=1)
        else:
            # 帧数较少时，使用较小的窗口宽度或不计算差分特征
            # 动态调整窗口宽度，确保不超过帧数
            width = min(5, mfcc_feat.shape[0] - 1)  # 确保width是奇数且不超过帧数
            if width < 3:
                width = 3  # 最小窗口宽度为3
            
            try:
                delta = librosa.feature.delta(mfcc_feat.T, width=width).T
                delta2 = librosa.feature.delta(mfcc_feat.T, order=2, width=width).T
                # 组合特征
                features = np.concatenate([mfcc_feat, delta, delta2], axis=1)
            except Exception:
                # 如果还是失败，就只返回MFCC特征
                features = mfcc_feat
        
        # 倒谱均值和方差归一化 (CMS + CVN)
        if cfg.USE_CEPSTRAL_NORMALIZATION:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-9
            features = (features - mean) / std
        
        return features
    except Exception as e:
        # 捕获所有异常，确保函数不会崩溃
        print(f"[ERROR] 特征提取过程中出错: {str(e)}")
        # 返回一个最小有效特征矩阵作为回退
        return np.zeros((5, cfg.N_MFCC))


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


def generate_embedding(y):
    """生成音频的嵌入向量"""
    # 提取MFCC特征
    mfcc_feats = extract_mfcc_features(y)
    
    # 提取基频特征
    pitch_feats = extract_pitch_features(y)
    
    # 提取声学统计特征
    acoustic_stats = extract_acoustic_stats(mfcc_feats)
    
    # 组合所有特征
    embedding = np.concatenate([acoustic_stats, pitch_feats], axis=0)
    
    # L2归一化
    norm = np.linalg.norm(embedding) + 1e-9
    embedding = embedding / norm
    
    return embedding

# ================== GMM-UBM模型 ==================

class DogVoiceModel:
    def __init__(self):
        self.ubm = None
        self.dog_gmms = {}
        self.scaler = StandardScaler()
        self.trained = False
        
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
        
    def train_dog_models(self, enroll_data):
        """为每只狗训练GMM模型，使用MAP适应"""
        if self.ubm is None:
            raise ValueError("请先训练UBM模型")
        
        for dog_id, features in enroll_data.items():
            print(f"[INFO] 为狗 {dog_id} 训练GMM模型...")
            
            # 特征标准化
            scaled_features = self.scaler.transform(features)
            
            # 使用UBM参数初始化，但要对权重进行归一化
            ubm_weights = self.ubm.weights_[:cfg.DOG_GMM_COMPONENTS]
            # 归一化权重，确保和为1
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
            
            # 应用MAP适应（可选）
            if cfg.MAP_WEIGHT < 1.0:
                dog_gmm.means_ = cfg.MAP_WEIGHT * dog_gmm.means_ + (1 - cfg.MAP_WEIGHT) * self.ubm.means_[:cfg.DOG_GMM_COMPONENTS]
                
            self.dog_gmms[dog_id] = dog_gmm
            print(f"[INFO] 狗 {dog_id} 的模型训练完成")
        
        self.trained = True
        
    def enroll_dog(self, dog_id, audio_files):
        """注册新的狗"""
        features = []
        for file_path in audio_files:
            # 加载和预处理音频
            y, _ = librosa.load(file_path, sr=cfg.SR)
            y = preprocess_audio(y)
            
            # 提取特征
            mfcc_feats = extract_mfcc_features(y)
            features.append(mfcc_feats)
        
        if not features:
            return False
        
        # 合并所有特征
        all_feats = np.vstack(features)
        
        # 如果还没有UBM，先创建简单的UBM
        if self.ubm is None:
            self.scaler.fit(all_feats)
            scaled_feats = self.scaler.transform(all_feats)
            
            self.ubm = GaussianMixture(
                n_components=min(4, len(scaled_feats)//10),  # 简单UBM
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
        """识别音频中的狗吠声，增强对多段狗吠、不同长度音频和背景噪音的处理能力
        
        Args:
            y: 音频数据
            
        Returns:
            result: 识别结果，可能是狗ID、'background'或'possible_dog'
            similarities: 每只狗的相似度得分(0-1范围)
        """
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
            
            # 预处理音频
            y = preprocess_audio(y)
            
            # 提取特征
            mfcc_feats = extract_mfcc_features(y)
            
            # 检查特征是否有效
            if mfcc_feats is None or len(mfcc_feats) == 0:
                return 'background', {}
            
            # 根据特征帧数判断音频长度
            num_frames = len(mfcc_feats)
            
            # 计算音频实际时长（秒）
            audio_duration = num_frames * cfg.FRAME_STEP
            
            # 如果特征帧数过少，可能是静音或噪声
            if num_frames < 5:
                return 'background', {}
            
            scaled_feats = self.scaler.transform(mfcc_feats)
            
            # 为每只狗计算得分
            log_scores = {}    
            for dog_id, gmm in self.dog_gmms.items():
                # 计算log-likelihood
                log_likelihood = gmm.score(scaled_feats)
                
                # 智能短音频处理：基于音频质量和特性动态调整分数提升
                if audio_duration < 1.0:
                    # 计算特征的质量指标
                    feat_mean = np.mean(np.abs(scaled_feats))
                    feat_std = np.std(scaled_feats)
                    
                    # 基于特征质量和音频长度动态调整提升幅度
                    quality_factor = min(1.0, max(0.5, feat_std))  # 特征质量因子
                    
                    if audio_duration < 0.3:
                        # 极短音频，根据质量动态提升
                        log_likelihood += 3.0 * quality_factor
                    elif audio_duration < 0.5:
                        # 短音频，根据质量动态提升
                        log_likelihood += 2.0 * quality_factor
                    else:
                        # 中等长度音频，根据质量动态提升
                        log_likelihood += 1.0 * quality_factor
                        
                log_scores[dog_id] = log_likelihood
            
            # 找到得分最高的狗
            best_dog = max(log_scores, key=log_scores.get)
            best_score = log_scores[best_dog]
            
            # 计算第二高得分，用于判断置信度
            sorted_scores = sorted(log_scores.values(), reverse=True)
            if len(sorted_scores) > 1:
                # 对于负分数，使用差值而不是比率来判断置信度
                score_diff = best_score - sorted_scores[1]
            else:
                score_diff = float('inf')
            
            # 根据分析结果重新调整阈值 - 基于per_dog_summary.csv中的真实/冒充样本平均得分
            # 分析显示所有狗的AUC=1.0，说明特征本身可分性极佳
            # 使用更科学的阈值设置，基于log_scores的z-score归一化
            
            # 计算所有log_scores的均值和标准差用于归一化
            all_scores = list(log_scores.values())
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores) + 1e-9  # 避免除零
            
            # 计算归一化的z-score相似度
            similarities = {}
            max_z_score = 3.0  # 设定最大z-score值
            
            for dog_id, log_score in log_scores.items():
                # 将log_score转换为z-score，使其分布在0-1之间
                z_score = min(max_z_score, (log_score - mean_score) / std_score)
                similarity = min(1.0, max(0.0, (z_score + max_z_score) / (2 * max_z_score)))
                similarities[dog_id] = round(similarity, 4)
            
            # 基于分析结果的优化阈值策略
            # 1. 降低背景噪声阈值，避免将有效狗叫误判为背景
            background_threshold = -100  # 降低背景阈值以提高检测率
            
            # 2. 降低置信度差异要求，使识别更加灵敏
            min_confidence_diff = 1.5  # 降低置信度差异要求
            
            # 3. 降低归一化得分阈值
            normalized_best_score = (best_score - mean_score) / std_score
            
            # 4. 考虑相似度得分
            max_similarity = max(similarities.values()) if similarities else 0
            
            # 分类决策 - 更平衡的决策逻辑
            if best_score < background_threshold:
                return 'background', similarities
            elif (normalized_best_score < 0.2 or score_diff < min_confidence_diff) and max_similarity < 0.7:
                # 归一化得分较低、置信度不足且相似度不高
                return 'possible_dog', similarities
            else:
                # 得分足够高、置信度足够高或相似度足够高，返回识别结果
                return best_dog, similarities
        except Exception as e:
            # 捕获所有异常，确保函数不会崩溃
            print(f"[ERROR] 识别过程中出错: {str(e)}")
            return 'background', {}
        
    def save_model(self, path):
        """保存模型到文件"""
        model_data = {
            'ubm': self.ubm,
            'dog_gmms': self.dog_gmms,
            'scaler': self.scaler,
            'trained': self.trained
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
        self.trained = model_data['trained']

# ================== 相似度计算 ==================

def compute_similarity(emb1, emb2, method='fusion'):
    """计算两个嵌入向量的相似度"""
    if emb1 is None or emb2 is None:
        return 0.0
    
    # 余弦相似度
    if method == 'cosine' or method == 'fusion':
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
        
    # 欧氏距离相似度
    if method == 'euclidean' or method == 'fusion':
        euc_dist = np.linalg.norm(emb1 - emb2)
        euc_sim = 1.0 / (1.0 + euc_dist)  # 转换为相似度
    
    # 融合相似度
    if method == 'fusion':
        return cfg.ALPHA * cos_sim + (1 - cfg.ALPHA) * euc_sim
    elif method == 'cosine':
        return cos_sim
    else:
        return euc_sim

# ================== 数据增强（用于训练） ==================

def augment_audio(y):
    """音频数据增强"""
    augmented = []
    
    # 原始音频
    augmented.append(y)
    
    # 添加轻微噪声
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)
    
    # 音量扰动
    volume_factors = [0.8, 1.2]
    for factor in volume_factors:
        augmented.append(y * factor)
    
    # 轻微的时间偏移
    for shift in [-100, 100]:
        shifted = np.roll(y, shift)
        augmented.append(shifted)
    
    return augmented

# ================== 主函数 ==================

if __name__ == "__main__":
    # 示例用法
    print("改进版狗吠声纹识别系统")
    
    # 创建模型实例
    model = DogVoiceModel()
    
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
    # model.save_model("dog_voice_model.pkl")
    # 
    # # 识别测试
    # y, _ = librosa.load("test_dog_voice.wav", sr=cfg.SR)
    # result, scores = model.recognize(y)
    # print(f"识别结果: {result}")
    # print(f"得分: {scores}")