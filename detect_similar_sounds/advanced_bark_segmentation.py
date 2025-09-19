# -*- coding: utf-8 -*-
"""
@Time    : 2024/9/18
@Author  : AI Assistant
@FileName: advanced_bark_segmentation.py
@Software: PyCharm
"""

"""
高级狗吠片段检测和裁剪工具
基于传统规则方法的精确狗吠分割系统
特点：自适应阈值、连续性检测、智能后处理规则
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from collections import deque

# 添加项目根目录到Python路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

class AdvancedBarkSegmenter:
    def __init__(self, 
                 target_sr=16000, 
                 frame_length=0.025,  # 25ms窗口
                 hop_length=0.01,     # 10ms步长
                 energy_window_ms=300,  # 能量统计窗口（增大以提高稳定性）
                 min_bark_duration=0.08,  # 最小狗吠时长80ms（降低以捕获更短的吠叫）
                 max_bark_duration=2,  # 最大狗吠时长2s（增加以避免分割过长的吠叫）
                 merge_gap_threshold=0.15,  # 合并间隙阈值150ms（降低以更好地合并连续吠叫）
                 short_gap_threshold=0.03,  # 连续吠叫短间隙30ms（降低以捕获极短间隙）
                 confidence_threshold=0.5):  # 置信度阈值（降低以提高检测灵敏度）
        """初始化高级狗吠分割器参数"""
        self.target_sr = target_sr
        self.frame_length = int(frame_length * target_sr)
        self.hop_length = int(hop_length * target_sr)
        self.energy_window_size = int(energy_window_ms / (hop_length * 1000))  # 能量统计窗口帧数
        self.min_bark_duration = min_bark_duration
        self.max_bark_duration = max_bark_duration
        self.merge_gap_threshold = merge_gap_threshold
        self.short_gap_threshold = short_gap_threshold
        self.confidence_threshold = confidence_threshold
        
    def preprocess_audio(self, y, sr):
        """预处理音频：转换为单声道、重采样、标准化"""
        # 转换为单声道
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
            
        # 重采样到目标采样率
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            
        # 音频标准化
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > 0:
            y = y / max_amplitude
            
        return y, self.target_sr
        
    def extract_features(self, y, sr):
        """提取音频特征：短时能量、过零率、频谱特征、MFCC等"""
        # 计算短时能量
        energy = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # 计算过零率
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # 计算频谱质心（高频内容指标）
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                           n_fft=self.frame_length, 
                                                           hop_length=self.hop_length)[0]
        
        # 计算频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                           n_fft=self.frame_length, 
                                                           hop_length=self.hop_length)[0]
        
        # 计算频谱滚降点
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, 
                                                         n_fft=self.frame_length, 
                                                         hop_length=self.hop_length)[0]
                                                         
        # 计算梅尔频率倒谱系数 (MFCC) - 对声音识别非常有效
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, 
                                   n_fft=self.frame_length, 
                                   hop_length=self.hop_length)
        
        # 计算色度特征 - 对音高敏感
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, 
                                           n_fft=self.frame_length, 
                                           hop_length=self.hop_length)
        
        # 确保所有特征长度一致
        min_length = min(len(energy), len(zcr), len(spectral_centroid), 
                         len(spectral_bandwidth), len(spectral_rolloff), 
                         mfcc.shape[1], chroma.shape[1])
        
        # 计算狗吠特征分数：基于典型的狗吠频率范围 (250Hz - 3kHz)
        # 计算频谱
        S = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length))
        # 获取频率轴
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
        # 找到狗吠频率范围的索引
        bark_freq_min = 250  # 狗吠最低频率
        bark_freq_max = 3000  # 狗吠最高频率
        bark_freq_indices = np.where((freqs >= bark_freq_min) & (freqs <= bark_freq_max))[0]
        # 计算每个帧中狗吠频率范围的能量占比
        bark_energy_ratio = np.zeros(min_length)
        for i in range(min_length):
            total_energy = np.sum(S[:, i] ** 2)
            if total_energy > 0:
                bark_energy = np.sum(S[bark_freq_indices, i] ** 2)
                bark_energy_ratio[i] = bark_energy / total_energy
        
        features = {
            'energy': energy[:min_length],
            'zcr': zcr[:min_length],
            'spectral_centroid': spectral_centroid[:min_length],
            'spectral_bandwidth': spectral_bandwidth[:min_length],
            'spectral_rolloff': spectral_rolloff[:min_length],
            'mfcc': mfcc[:, :min_length],  # MFCC特征
            'chroma': chroma[:, :min_length],  # 色度特征
            'bark_energy_ratio': bark_energy_ratio,  # 狗吠频率范围内能量占比
            'times': librosa.frames_to_time(range(min_length), sr=sr, hop_length=self.hop_length)
        }
        
        return features
        
    def calculate_adaptive_threshold(self, energy):
        """计算自适应能量阈值：基于背景能量统计和动态调整"""
        # 使用滑动窗口计算背景能量统计
        energy_window = deque(maxlen=self.energy_window_size)
        thresholds = np.zeros_like(energy)
        
        # 初始窗口填充
        initial_window = min(self.energy_window_size, len(energy))
        energy_window.extend(energy[:initial_window])
        
        # 记录最近的能量峰值，用于动态调整阈值
        peak_history = deque(maxlen=5)  # 记录最近5个峰值
        
        # 记录活动历史，用于检测信号的非平稳性
        activity_history = deque(maxlen=20)  # 记录最近20帧的活动状态
        
        for i in range(len(energy)):
            if i >= initial_window:
                energy_window.append(energy[i])
                
            # 计算当前窗口的统计信息
            window_energy = np.array(energy_window)
            # 使用中位数而不是均值，提高对突发噪声的鲁棒性
            median_energy = np.median(window_energy)
            std_energy = np.std(window_energy)
            
            # 更新峰值历史
            if i > 0 and i < len(energy) - 1:
                if energy[i] > energy[i-1] and energy[i] > energy[i+1]:
                    peak_history.append(energy[i])
            
            # 阈值 = 中位数 + N倍标准差
            # 降低n_factor的最小值以提高检测灵敏度
            n_factor = max(1.0, 2.5 - min(1.2, median_energy * 100))
            
            # 检查信号是否处于非平稳状态
            is_high_energy = energy[i] > median_energy + 2 * std_energy
            activity_history.append(1 if is_high_energy else 0)
            
            # 如果最近有活动，降低阈值以保持检测连续性
            recent_activity = np.mean(activity_history) if len(activity_history) > 0 else 0
            if recent_activity > 0.3:  # 如果最近30%的帧是活动的
                n_factor = max(0.8, n_factor * 0.8)  # 降低阈值因子
            
            # 如果有峰值历史，使用峰值的一部分作为额外参考
            if peak_history:
                avg_peak = np.mean(peak_history)
                # 动态调整阈值，确保能捕获到相似强度的声音
                thresholds[i] = min(median_energy + n_factor * std_energy, avg_peak * 0.5)  # 使用峰值的50%作为上限，降低以捕获更多
            else:
                thresholds[i] = median_energy + n_factor * std_energy
            
        # 平滑阈值曲线，避免突变
        smoothed_thresholds = np.zeros_like(thresholds)
        smooth_window = min(5, len(thresholds) // 10)  # 平滑窗口大小
        if smooth_window > 1:
            for i in range(len(thresholds)):
                start = max(0, i - smooth_window // 2)
                end = min(len(thresholds), i + smooth_window // 2 + 1)
                smoothed_thresholds[i] = np.mean(thresholds[start:end])
        else:
            smoothed_thresholds = thresholds
        
        return smoothed_thresholds
        
    def detect_candidate_segments(self, y, sr, features):
        """基于能量阈值和多特征融合分析检测候选狗吠片段"""
        energy = features['energy']
        zcr = features['zcr']
        bark_energy_ratio = features['bark_energy_ratio']
        times = features['times']
        
        # 计算自适应阈值
        adaptive_thresholds = self.calculate_adaptive_threshold(energy)
        
        # 生成初始活动帧掩码
        is_active = energy > adaptive_thresholds
        
        # 为连续快速吠叫增加特殊处理：检测到活动后，降低阈值一小段时间
        for i in range(1, len(is_active)-1):
            if is_active[i] and is_active[i-1]:
                lookahead_frames = min(10, len(is_active) - i - 1)  # 向前看10帧
                for j in range(1, lookahead_frames+1):
                    if energy[i+j] > adaptive_thresholds[i+j] * 0.7:  # 更低的阈值以捕获连续吠叫
                        is_active[i+j] = True
        
        # 使用多特征融合计算置信度分数
        # 1. 能量置信度
        energy_confidence = np.clip((energy - adaptive_thresholds) / (np.max(energy) - adaptive_thresholds + 1e-8), 0, 1)
        
        # 2. 过零率置信度 - 狗吠通常有较高的过零率
        mean_zcr = np.mean(zcr)
        std_zcr = np.std(zcr)
        zcr_threshold_low = mean_zcr + 0.3 * std_zcr  # 降低下限以提高灵敏度
        zcr_threshold_high = mean_zcr + 2.5 * std_zcr
        zcr_confidence = np.zeros_like(zcr)
        zcr_confidence[zcr < zcr_threshold_low] = 0.3  # 过低的过零率可能不是狗吠
        zcr_confidence[(zcr >= zcr_threshold_low) & (zcr <= zcr_threshold_high)] = 0.8  # 中等范围很可能是狗吠
        zcr_confidence[zcr > zcr_threshold_high] = 0.5  # 过高的过零率可能是噪声
        
        # 3. 全局过零率检查
        # 注意：不同大小的狗，其狗吠的过零率可能有较大差异
        mean_zcr_global = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        std_zcr_global = np.std(librosa.feature.zero_crossing_rate(y)[0])
        
        # 对每个帧应用全局过零率检查
        for i in range(len(zcr)):
            if zcr[i] < mean_zcr_global - 0.5 * std_zcr_global or zcr[i] > mean_zcr_global + 2.5 * std_zcr_global:
                zcr_confidence[i] *= 0.3  # 降低不符合全局特征的帧的置信度
        
        # 3. 狗吠频率范围置信度
        bark_confidence = np.clip(bark_energy_ratio * 2 - 0.5, 0, 1)  # 映射到0-1范围，增强区分度
        
        # 4. 频谱特征置信度（这里简化处理）
        spectral_confidence = np.ones_like(energy)
        
        # 融合所有置信度，使用加权平均
        confidence = (0.4 * energy_confidence + 
                     0.25 * zcr_confidence + 
                     0.3 * bark_confidence + 
                     0.05 * spectral_confidence)
        
        # 检测片段边界
        candidate_segments = []
        current_start = None
        current_confidence_sum = 0
        current_frame_count = 0
        
        for i in range(len(is_active)):
            if is_active[i] and current_start is None:
                # 开始新片段
                current_start = i
                current_confidence_sum = confidence[i]
                current_frame_count = 1
                
            elif not is_active[i] and current_start is not None:
                # 片段结束
                segment_duration = (i - current_start) * self.hop_length / sr
                avg_confidence = current_confidence_sum / current_frame_count
                
                # 只保留满足最小时长和置信度要求的片段
                if (segment_duration >= self.min_bark_duration and 
                    avg_confidence >= self.confidence_threshold):
                    start_time = times[current_start]
                    end_time = times[i-1]
                    candidate_segments.append((start_time, end_time, avg_confidence))
                    
                current_start = None
                current_confidence_sum = 0
                current_frame_count = 0
                
            elif is_active[i]:
                # 片段延续
                current_confidence_sum += confidence[i]
                current_frame_count += 1
        
        # 处理最后一个片段
        if current_start is not None:
            segment_duration = (len(is_active) - current_start) * self.hop_length / sr
            avg_confidence = current_confidence_sum / current_frame_count
            
            if (segment_duration >= self.min_bark_duration and 
                avg_confidence >= self.confidence_threshold):
                start_time = times[current_start]
                end_time = times[-1]
                candidate_segments.append((start_time, end_time, avg_confidence))
                
        return candidate_segments, adaptive_thresholds
        
    def refine_segments(self, y, sr, segments):
        """优化片段边界并应用后处理规则"""
        if not segments:
            return []
            
        # 按起始时间排序
        segments = sorted(segments, key=lambda x: x[0])
        
        # 应用合并规则
        merged_segments = []
        for seg in segments:
            start, end, confidence = seg
            
            if not merged_segments:
                merged_segments.append([start, end, confidence])
            else:
                last_start, last_end, last_conf = merged_segments[-1]
                gap = start - last_end
                
                # 合并过短间隙
                if gap <= self.merge_gap_threshold:
                    # 特殊处理连续快速吠叫
                    if gap <= self.short_gap_threshold:
                        # 对于极短间隙，保留间隙但合并为一个连续吠叫事件
                        merged_segments[-1][1] = end  # 扩展结束时间
                        merged_segments[-1][2] = max(last_conf, confidence)  # 取较高置信度
                    else:
                        # 对于中等间隙，合并为一个片段
                        merged_segments[-1][1] = end
                        merged_segments[-1][2] = (last_conf + confidence) / 2  # 平均置信度
                else:
                    merged_segments.append([start, end, confidence])
        
        # 处理过长片段并进行特征验证
        final_segments = []
        for seg in merged_segments:
            start, end, confidence = seg
            duration = end - start
            
            # 如果片段过长，尝试分割并验证
            if duration > self.max_bark_duration:
                # 计算需要分割成多少段
                num_segments = int(np.ceil(duration / self.max_bark_duration))
                segment_duration = duration / num_segments
                
                # 执行分割并验证每个子段
                for i in range(num_segments):
                    seg_start = start + i * segment_duration
                    seg_end = min(seg_start + segment_duration, end)
                    
                    # 确保每个子段满足最小时长
                    if seg_end - seg_start >= self.min_bark_duration:
                        # 提取子段音频进行验证
                        seg_start_idx = max(0, int(seg_start * sr))
                        seg_end_idx = min(len(y), int(seg_end * sr))
                        seg_audio = y[seg_start_idx:seg_end_idx]
                        
                        # 验证子段是否为狗吠
                        is_valid, seg_confidence = self._validate_bark_segment(seg_audio, sr, y)
                        
                        if is_valid:
                            # 合并原始置信度和验证置信度
                            combined_confidence = (confidence + seg_confidence) / 2
                            final_segments.append((seg_start, seg_end, combined_confidence))
            else:
                # 对正常长度的片段进行验证
                seg_start_idx = max(0, int(start * sr))
                seg_end_idx = min(len(y), int(end * sr))
                seg_audio = y[seg_start_idx:seg_end_idx]
                
                # 验证片段是否为狗吠
                is_valid, seg_confidence = self._validate_bark_segment(seg_audio, sr, y)
                
                if is_valid:
                    # 合并原始置信度和验证置信度
                    combined_confidence = (confidence + seg_confidence) / 2
                    final_segments.append((start, end, combined_confidence))
        
        # 基于置信度再次过滤
        if final_segments:
            final_segments.sort(key=lambda x: x[2], reverse=True)
            avg_confidence = np.mean([s[2] for s in final_segments])
            min_confidence = max(0.3, avg_confidence * 0.6)
            final_segments = [s for s in final_segments if s[2] >= min_confidence]
        
        # 转换为时间戳列表格式
        timestamp_segments = [(start, end) for start, end, _ in final_segments]
        
        return timestamp_segments
        
    def precise_boundary_refinement(self, y, sr, segments):
        """精确细化片段边界"""
        refined_segments = []
        
        for start_time, end_time in segments:
            # 转换为样本索引
            start_idx = max(0, int(start_time * sr))
            end_idx = min(len(y), int(end_time * sr))
            
            # 提取片段
            segment = y[start_idx:end_idx]
            
            # 计算希尔伯特包络作为更精确的能量包络
            if len(segment) > 10:
                try:
                    envelope = np.abs(signal.hilbert(segment))
                    
                    # 自适应阈值：使用包络的统计特性
                    # 降低边界阈值以更好地捕获弱信号尾部
                    mean_env = np.mean(envelope)
                    std_env = np.std(envelope)
                    boundary_threshold = max(mean_env * 0.05, std_env * 0.3)
                    
                    # 增加尾部检测增强：在找到主要结束点后，再检查一小段时间
                    # 这有助于捕获狗吠的完整尾部声音
                    
                    # 精确定位起始点
                    precise_start_offset = 0
                    for i, amp in enumerate(envelope):
                        if amp > boundary_threshold:
                            # 向前延伸10ms作为预触发（增加以捕获完整的狗吠起始）
                            precise_start_offset = max(0, i - int(0.01 * sr))
                            break
                    
                    # 精确定位结束点
                    precise_end_offset = len(envelope)
                    
                    # 首先找到主要结束点
                    main_end = len(envelope)
                    for i in range(len(envelope) - 1, -1, -1):
                        if envelope[i] > boundary_threshold:
                            main_end = i
                            break
                    
                    # 增加尾部检测增强：在主要结束点后，继续检查声音衰减部分
                    # 这有助于捕获狗吠的完整尾部声音，即使音量逐渐降低
                    extended_end = main_end
                    tail_search_win = int(0.1 * sr)  # 向后搜索100ms
                    tail_start = min(main_end + int(0.01 * sr), len(envelope))  # 从主要结束点稍后开始
                    tail_end = min(tail_start + tail_search_win, len(envelope))
                    
                    # 计算尾部区域的特征
                    if tail_end > tail_start:
                        tail_envelope = envelope[tail_start:tail_end]
                        tail_mean = np.mean(tail_envelope)
                        tail_std = np.std(tail_envelope)
                        tail_threshold = max(tail_mean * 0.1, tail_std * 0.3)
                        
                        # 在尾部区域寻找声音
                        for i in range(tail_end - 1, tail_start - 1, -1):
                            if envelope[i] > tail_threshold:
                                extended_end = i
                                break
                    
                    # 设置最终结束点，包含30ms的后触发
                    precise_end_offset = min(len(envelope), extended_end + int(0.03 * sr))
                    
                    # 转换回时间
                    precise_start = start_time + precise_start_offset / sr
                    precise_end = start_time + precise_end_offset / sr
                    
                    # 确保修改后的片段仍然满足时长要求
                    if precise_end - precise_start >= self.min_bark_duration:
                        refined_segments.append((precise_start, precise_end))
                        continue
                except:
                    # 如果希尔伯特变换失败，使用原始边界
                    pass
            
            # 如果细化失败，使用原始边界
            refined_segments.append((start_time, end_time))
        
        return refined_segments
        
    def detect_bark_segments(self, y, sr=None):
        """主函数：检测狗吠片段"""
        # 检查输入类型
        if isinstance(y, str):
            # 如果传入的是文件路径，则加载音频
            y, sr = librosa.load(y, sr=None)
            
        # 预处理音频
        y_processed, sr_processed = self.preprocess_audio(y, sr)
        
        # 提取特征
        features = self.extract_features(y_processed, sr_processed)
        
        # 检测候选片段
        candidate_segments, _ = self.detect_candidate_segments(y_processed, sr_processed, features)
        
        # 应用后处理规则
        refined_segments = self.refine_segments(y_processed, sr_processed, candidate_segments)
        
        # 精确细化边界
        final_segments = self.precise_boundary_refinement(y_processed, sr_processed, refined_segments)
        
        # 转换为字典格式，并进行最终的特征验证和置信度调整
        segment_dicts = []
        for start, end in final_segments:
            # 查找对应置信度
            confidence = 0.5  # 默认置信度
            for seg in candidate_segments:
                if seg[0] <= start and seg[1] >= end:
                    confidence = seg[2]
                    break
            
            # 额外的置信度调整：基于片段特征
            start_idx = max(0, int(start * sr_processed))
            end_idx = min(len(y_processed), int(end * sr_processed))
            segment_audio = y_processed[start_idx:end_idx]
            
            # 执行最终的片段验证
            if len(segment_audio) > 10:
                try:
                    # 验证片段是否为狗吠
                    is_valid, validation_confidence = self._validate_bark_segment(segment_audio, sr_processed, y_processed)
                    
                    if is_valid:
                        # 合并原始置信度和验证置信度
                        confidence = (confidence + validation_confidence) / 2
                        
                        # 再次检查狗吠频率范围内的能量占比
                        S = librosa.feature.melspectrogram(y=segment_audio, sr=sr_processed, n_mels=128)
                        bark_freq_min = 250
                        bark_freq_max = 3000
                        mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr_processed/2)
                        bark_freq_indices = np.where((mel_freqs >= bark_freq_min) & (mel_freqs <= bark_freq_max))[0]
                        
                        if len(bark_freq_indices) > 0:
                            total_energy = np.sum(np.abs(S) ** 2)
                            bark_energy = np.sum(np.abs(S[bark_freq_indices, :]) ** 2)
                            bark_ratio = bark_energy / total_energy if total_energy > 0 else 0
                            
                            # 根据狗吠频率占比调整置信度
                            confidence = confidence * (0.5 + 0.5 * bark_ratio)  # 最高可提升50%
                except:
                    pass
            
            segment_dicts.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start,
                'confidence': confidence
            })
        
        # 按置信度排序，便于后续处理
        segment_dicts.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 应用最终的置信度阈值过滤
        if segment_dicts:
            # 动态确定置信度阈值，结合全局平均值和分布特性
            confidences = [s['confidence'] for s in segment_dicts]
            mean_confidence = np.mean(confidences)
            median_confidence = np.median(confidences)
            std_confidence = np.std(confidences)
            
            # 使用中位数和标准差的组合来确定更合理的阈值
            min_confidence_threshold = max(0.35, min(median_confidence, mean_confidence - 0.5 * std_confidence))
            
            # 保留置信度高于阈值的片段
            segment_dicts = [s for s in segment_dicts if s['confidence'] >= min_confidence_threshold]
            
            # 额外的过滤：移除明显不合理的片段（过短或过长）
            # 考虑到已经有基本的长度过滤，这里使用更严格的异常值检测
            if len(segment_dicts) > 1:
                durations = [s['duration'] for s in segment_dicts]
                mean_duration = np.mean(durations)
                std_duration = np.std(durations)
                
                segment_dicts = [s for s in segment_dicts if 
                                mean_duration - 2 * std_duration <= s['duration'] <= mean_duration + 2 * std_duration]
        
        return segment_dicts
        
    def _bark_pattern_classifier(self, segment_audio, sr):
        """基于狗吠的声学模式进行分类，识别典型的狗吠模式"""
        try:
            # 计算MFCC特征，这是声音识别中最有效的特征之一
            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            
            # 计算MFCC的一阶和二阶差分
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # 提取统计特征
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
            delta_mfcc_std = np.std(delta_mfcc, axis=1)
            delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)
            
            # 狗吠的典型模式规则 - 更精细的调整
            rules = []
            
            # 1. 第一MFCC系数范围（与音高相关）
            rule1 = -45 <= mfcc_mean[0] <= 25  # 稍微扩大范围减少漏报
            rules.append(rule1)
            
            # 2. 第二MFCC系数范围
            rule2 = -40 <= mfcc_mean[1] <= 30
            rules.append(rule2)
            
            # 3. 高阶MFCC的标准差（与音色复杂度相关）
            high_order_std = np.mean(mfcc_std[4:])
            rule3 = high_order_std > 4  # 稍微降低标准
            rules.append(rule3)
            
            # 4. 差分MFCC的均值（与声音变化率相关）
            rule4 = -3 <= np.mean(delta_mfcc_mean) <= 3  # 稍微放宽标准
            rules.append(rule4)
            
            # 5. 差分MFCC的标准差（衡量变化稳定性）
            rule5 = np.mean(delta_mfcc_std) < 10
            rules.append(rule5)
            
            # 6. 二阶差分MFCC的均值（衡量加速度变化）
            rule6 = -5 <= np.mean(delta2_mfcc_mean) <= 5
            rules.append(rule6)
            
            # 7. MFCC系数间的相关性（相邻系数应有较高相关性）
            if len(mfcc_mean) > 3:
                correlation = np.corrcoef(mfcc_mean[:4])[0, 1]
                rule7 = abs(correlation) > 0.4
                rules.append(rule7)
            
            # 8. 能量变化模式检查
            rms = librosa.feature.rms(y=segment_audio)[0]
            rule8 = len(rms) > 0 and np.std(rms) > 0.05
            rules.append(rule8)
            
            # 综合判断：满足至少4个规则，且必须包含规则1和规则3（最重要的特征）
            satisfied_rules = sum(rules)
            return satisfied_rules >= 4 and rule1 and rule3
        except:
            # 异常情况下返回False
            return False
        
    def _validate_bark_segment(self, segment_audio, sr, y=None):
        """
        验证片段是否为真实的狗吠声
        使用多种特征进行综合判断
        """
        # 初始化判断标志和置信度分数
        is_bark = True
        confidence_score = 0.5
        
        # 1. 使用模式分类器进行初步判断
        pattern_match = self._bark_pattern_classifier(segment_audio, sr)
        if not pattern_match:
            confidence_score -= 0.3
        else:
            confidence_score += 0.3  # 增加权重以提高模式匹配重要性
        
        # 2. 狗吠频率范围内的能量占比 - 更严格的阈值
        try:
            S = librosa.feature.melspectrogram(y=segment_audio, sr=sr, n_mels=128)
            bark_freq_min = 250
            bark_freq_max = 3000
            mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)
            bark_freq_indices = np.where((mel_freqs >= bark_freq_min) & (mel_freqs <= bark_freq_max))[0]
            
            if len(bark_freq_indices) > 0:
                total_energy = np.sum(np.abs(S) ** 2)
                bark_energy = np.sum(np.abs(S[bark_freq_indices, :]) ** 2)
                bark_ratio = bark_energy / total_energy if total_energy > 0 else 0
                
                # 根据狗吠频率占比调整置信度
                if bark_ratio > 0.4:  # 提高阈值至0.4
                    confidence_score += (bark_ratio - 0.4) * 0.8  # 增加调整系数
                
                # 过低的占比直接降低置信度
                if bark_ratio < 0.3:
                    confidence_score -= 0.4  # 增加惩罚力度
        except:
            pass
        
        # 3. 检查频谱中心频率
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroid)
            
            # 狗吠声的频谱中心通常在300-5000Hz范围内
            if mean_centroid < 300 or mean_centroid > 5000:
                confidence_score -= 0.3  # 增加惩罚力度
            else:
                # 最佳狗吠中心频率范围是800-2500Hz
                if 800 <= mean_centroid <= 2500:
                    confidence_score += 0.2  # 增加奖励
        except:
            pass
        
        # 4. 检查过零率 - 更严格的全局比较
        try:
            zcr = librosa.feature.zero_crossing_rate(segment_audio)[0]
            mean_zcr = np.mean(zcr)
            
            # 检查过零率是否符合狗吠特性
            if y is not None:
                mean_zcr_global = np.mean(librosa.feature.zero_crossing_rate(y)[0])
                std_zcr_global = np.std(librosa.feature.zero_crossing_rate(y)[0])
                
                # 更严格的范围，减少与背景差异过大的误报
                if mean_zcr < mean_zcr_global - 0.3 * std_zcr_global or mean_zcr > mean_zcr_global + 2.0 * std_zcr_global:
                    confidence_score -= 0.3  # 增加惩罚
        except:
            pass
        
        # 5. 检查频谱带宽和丰富度 - 更精细的范围
        try:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)[0]
            mean_bandwidth = np.mean(spectral_bandwidth)
            
            # 狗吠声通常具有一定的带宽范围
            if mean_bandwidth < 200 or mean_bandwidth > 3500:
                confidence_score -= 0.2  # 调整惩罚力度
            
            # 计算频谱平整度（新增特征）
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=segment_audio)[0])
            
            # 狗吠声通常具有中等的频谱平整度
            if spectral_flatness < 0.01 or spectral_flatness > 0.3:
                confidence_score -= 0.15  # 添加平整度检查惩罚
        except:
            pass
        
        # 6. 检查音频的动态范围和峰值特性
        try:
            # 计算音频的峰值与均值之比
            peak_to_mean = np.max(np.abs(segment_audio)) / (np.mean(np.abs(segment_audio)) + 1e-10)
            
            # 狗吠通常具有明显的峰值
            if peak_to_mean < 3:  # 提高阈值至3
                confidence_score -= 0.15  # 增加惩罚力度
            
            # 计算音频的信噪比估计
            rms = np.sqrt(np.mean(np.square(segment_audio)))
            if rms < 0.02:
                confidence_score -= 0.1  # 低能量片段惩罚
        except:
            pass
        
        # 归一化置信度分数
        confidence_score = max(0, min(1, confidence_score))
        
        # 基于置信度做出最终判断 - 提高阈值减少误报
        is_bark = is_bark and (confidence_score >= 0.45)
        
        return is_bark, confidence_score
        
    def visualize_results(self, y, sr, segments, title="高级狗吠片段检测结果"):
        """可视化检测结果"""
        plt.figure(figsize=(15, 8))
        
        # 显示波形
        librosa.display.waveshow(y, sr=sr, alpha=0.7)
        
        # 标记检测到的片段
        for i, (start, end) in enumerate(segments):
            plt.axvspan(start, end, alpha=0.3, color='red', label=f'狗吠片段 {i+1}' if i == 0 else "")
        
        plt.title(title)
        plt.xlabel("时间 (秒)")
        plt.ylabel("幅度")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def extract_and_save_segments(self, audio_path, segments, output_dir=None):
        """从音频文件中提取并保存狗吠片段"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=None)
        
        if len(segments) == 0:
            print("未检测到任何狗吠片段")
            return []
        
        # 统计信息
        print(f"检测到 {len(segments)} 个狗吠片段:")
        total_duration = 0
        short_segments = 0  # <0.5s
        medium_segments = 0  # 0.5-1.0s
        long_segments = 0  # >1.0s
        
        for i, seg in enumerate(segments):
            start = seg['start_time']
            end = seg['end_time']
            duration = seg['duration']
            confidence = seg['confidence']
            
            total_duration += duration
            
            if duration < 0.5:
                short_segments += 1
            elif duration < 1.0:
                medium_segments += 1
            else:
                long_segments += 1
                
            print(f"  片段 {i+1}: {start:.3f}s - {end:.3f}s (持续时间: {duration:.3f}s, 置信度: {confidence:.2f})")
        
        print(f"\n统计信息:")
        print(f"  总持续时间: {total_duration:.3f}s")
        print(f"  短片段 (<0.5s): {short_segments}个")
        print(f"  中等片段 (0.5-1.0s): {medium_segments}个")
        print(f"  长片段 (>1.0s): {long_segments}个")
        
        # 确定输出目录
        if output_dir is None:
            base_dir = os.path.dirname(audio_path)
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = os.path.join(base_dir, f"{filename}_bark_segments_advanced")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 预处理后的音频
        y_processed, sr_processed = self.preprocess_audio(y, sr)
        
        # 保存片段
        saved_files = []
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 保存时间戳列表
        timestamp_path = os.path.join(output_dir, f"{base_name}_bark_timestamps.txt")
        with open(timestamp_path, 'w', encoding='utf-8') as f:
            f.write("# 狗吠片段时间戳 (开始时间(秒), 结束时间(秒), 持续时间(秒), 置信度)\n")
        
        for i, seg in enumerate(segments):
            start = seg['start_time']
            end = seg['end_time']
            duration = seg['duration']
            confidence = seg['confidence']
            
            # 转换为样本索引
            start_idx = max(0, int(start * sr_processed))
            end_idx = min(len(y_processed), int(end * sr_processed))
            
            # 提取片段
            segment = y_processed[start_idx:end_idx]
            
            # 保存片段
            segment_filename = f"{base_name}_bark_{i+1:02d}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            
            sf.write(segment_path, segment, sr_processed)
            saved_files.append(segment_path)
            print(f"已保存片段: {segment_path}")
            
            # 写入时间戳
            with open(timestamp_path, 'a', encoding='utf-8') as f:
                f.write(f"{start:.3f}, {end:.3f}, {duration:.3f}, {confidence:.2f}\n")
        
        print(f"已保存时间戳列表到: {timestamp_path}")
        
        return saved_files


# 命令行使用示例
if __name__ == "__main__":
    # 示例用法
    import argparse
    import os
    import time
    
    parser = argparse.ArgumentParser(description='Advanced Bark Segmentation')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for bark segments')
    parser.add_argument('--visualize', action='store_true', help='Visualize the detection results')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # 确保输入文件存在
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file '{args.audio_path}' not found.")
        exit(1)
    
    # 创建输出目录（如果指定）
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    print("Initializing Advanced Bark Segmenter...")
    
    # 初始化分割器并执行分割
    segmenter = AdvancedBarkSegmenter(
        target_sr=16000,
        min_bark_duration=0.08,  # 降低最小狗吠时长至0.08秒
        max_bark_duration=2.5,   # 增加最大狗吠时长至2.5秒
        merge_gap_threshold=0.15,  # 降低合并间隙阈值至0.15秒
        energy_window_ms=30,    # 增大能量窗口至30帧（约300ms）
        short_gap_threshold=0.03, # 降低短间隙阈值至0.03秒
        confidence_threshold=0.5  # 降低置信度阈值至0.5
    )
    
    # 执行分割并计时
    print(f"Processing audio file: {args.audio_path}")
    start_time = time.time()
    bark_segments = segmenter.detect_bark_segments(args.audio_path)
    processing_time = time.time() - start_time
    
    # 输出检测结果统计
    print(f"Detection completed in {processing_time:.2f} seconds")
    print(f"Found {len(bark_segments)} bark segments")
    
    if args.debug and bark_segments:
        print("Segment details:")
        for i, segment in enumerate(bark_segments):
            start_time = segment['start_time']
            end_time = segment['end_time']
            duration = end_time - start_time
            confidence = segment.get('confidence', 0)
            print(f"  Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s), Confidence: {confidence:.2f}")
    
    # 可视化结果
    if args.visualize:
        print("Generating visualization...")
        segmenter.visualize_results(args.audio_path, bark_segments)
    
    # 提取并保存片段
    saved_files = []
    if args.output_dir:
        print(f"Saving bark segments to {args.output_dir}...")
        saved_files = segmenter.extract_and_save_segments(args.audio_path, bark_segments, args.output_dir)
    
    print("Advanced bark segmentation completed successfully!")
    
    # 示例命令:
    # python advanced_bark_segmentation.py audio.wav --output_dir segments --visualize --debug
    # python advanced_bark_segmentation.py "path/to/your/audio.wav" --output_dir "output_segments" --visualize

"""
使用示例:

python detect_similar_sounds/advanced_bark_segmentation.py D:\kend\myPython\speechDog-master\youtube_wav\bark_segmentation\processed_template_preserving\outdoor_braking_clean_preserving.wav

"""