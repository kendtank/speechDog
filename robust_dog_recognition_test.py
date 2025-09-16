# -*- coding: utf-8 -*-
"""
狗吠声纹识别系统 - 量产级鲁棒性测试
测试不同场景下的识别性能：
- 多段狗吠处理
- 不同长度音频识别
- 背景噪声处理
- 短音频(0.2s)识别
- 展示0-1范围的相似度分数
"""

import os
import numpy as np
import librosa
import time
import glob
from enhanced_dog_voice_recognition import DogVoiceModel

class RobustDogRecognitionTester:
    def __init__(self, model_path="dog_voice_model.pkl"):
        """初始化测试器"""
        self.model = DogVoiceModel()
        try:
            self.model.load_model(model_path)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def load_audio(self, file_path, sr=16000):
        """加载音频文件"""
        try:
            y, sr = librosa.load(file_path, sr=sr)
            return y
        except Exception as e:
            print(f"❌ 音频加载失败 ({file_path}): {e}")
            return None
    
    def split_into_segments(self, y, segment_duration=0.5, sr=16000):
        """将音频分割成多个短片段"""
        segment_length = int(segment_duration * sr)
        segments = []
        start = 0
        while start + segment_length <= len(y):
            segments.append(y[start:start + segment_length])
            start += segment_length
        
        # 如果最后一段不够长，也保留
        if start < len(y):
            segments.append(y[start:])
            
        return segments
    
    def detect_multiple_barks(self, y, sr=16000, min_bark_duration=0.2, min_silence_duration=0.1):
        """检测音频中的多段狗吠声"""
        # 使用RMS能量检测语音活动
        frame_length = int(0.025 * sr)
        hop_length = int(0.01 * sr)
        
        # 计算RMS能量
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 简单阈值检测
        threshold = 0.005 * np.max(rms)
        voice_frames = rms > threshold
        
        # 转换为时间点
        voice_times = librosa.frames_to_time(np.where(voice_frames)[0], sr=sr, hop_length=hop_length)
        
        if len(voice_times) == 0:
            return []
        
        # 合并相邻的语音段
        segments = []
        start = voice_times[0]
        prev = voice_times[0]
        
        for t in voice_times[1:]:
            if t - prev > min_silence_duration:
                # 如果间隔超过最小静音时长，视为新的狗吠段
                if prev - start >= min_bark_duration:
                    segments.append((start, prev))
                start = t
            prev = t
        
        # 添加最后一个段
        if prev - start >= min_bark_duration:
            segments.append((start, prev))
        
        # 转换为音频片段
        audio_segments = []
        for start_time, end_time in segments:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segments.append(y[start_sample:end_sample])
        
        return audio_segments
    
    def recognize_single_file(self, file_path):
        """识别单个音频文件"""
        print(f"\n=== 测试文件: {os.path.basename(file_path)} ===")
        
        # 加载音频
        y = self.load_audio(file_path)
        if y is None:
            return None, None
        
        # 获取音频时长
        duration = librosa.get_duration(y=y, sr=16000)
        print(f"⏱️ 音频时长: {duration:.2f}秒")
        
        # 检测多段狗吠
        bark_segments = self.detect_multiple_barks(y)
        if len(bark_segments) > 0:
            print(f"🔍 检测到 {len(bark_segments)} 段狗吠声")
            
            # 对每段狗吠进行识别
            segment_results = []
            for i, segment in enumerate(bark_segments):
                seg_duration = librosa.get_duration(y=segment, sr=16000)
                print(f"  段 {i+1}: 时长 {seg_duration:.2f}秒")
                result, similarities = self.model.recognize(segment)
                segment_results.append((result, similarities))
                print(f"  识别结果: {result}")
                if similarities:
                    print("  相似度分数:", {k: f"{v:.3f}" for k, v in similarities.items()})
            
            # 综合多段识别结果
            final_result = self.combine_segment_results(segment_results)
            return final_result, segment_results
        else:
            # 对整个音频进行识别
            start_time = time.time()
            result, similarities = self.model.recognize(y)
            process_time = time.time() - start_time
            
            print(f"⚡ 处理时间: {process_time:.3f}秒")
            print(f"📝 识别结果: {result}")
            if similarities:
                print("📊 相似度分数:", {k: f"{v:.3f}" for k, v in similarities.items()})
            
            return result, similarities
    
    def combine_segment_results(self, segment_results):
        """综合多段狗吠的识别结果"""
        # 简单的多数投票策略
        dog_votes = {}
        for result, _ in segment_results:
            if result not in ['background', 'possible_dog']:
                if result in dog_votes:
                    dog_votes[result] += 1
                else:
                    dog_votes[result] = 1
        
        if dog_votes:
            # 返回得票最多的狗
            return max(dog_votes, key=dog_votes.get)
        else:
            # 如果没有明确的狗吠识别结果，返回出现最多的结果类型
            type_votes = {'background': 0, 'possible_dog': 0}
            for result, _ in segment_results:
                if result in type_votes:
                    type_votes[result] += 1
            
            if type_votes['possible_dog'] > type_votes['background']:
                return 'possible_dog'
            else:
                return 'background'
    
    def batch_test(self, test_dir, file_pattern="*.WAV"):
        """批量测试多个文件"""
        print(f"\n🚀 开始批量测试 - 目录: {test_dir}")
        
        # 获取所有测试文件
        search_path = os.path.join(test_dir, file_pattern)
        test_files = glob.glob(search_path)
        
        if not test_files:
            print(f"❌ 没有找到匹配的文件: {search_path}")
            return
        
        print(f"📋 找到 {len(test_files)} 个测试文件")
        
        # 统计结果
        total_files = len(test_files)
        correct_count = 0
        error_count = 0
        background_count = 0
        possible_dog_count = 0
        total_time = 0
        
        for file_path in test_files:
            # 尝试从文件名提取真实标签
            file_name = os.path.basename(file_path)
            true_label = None
            for dog_id in self.model.dog_gmms.keys():
                if dog_id in file_name:
                    true_label = dog_id
                    break
            
            start_time = time.time()
            result, _ = self.recognize_single_file(file_path)
            process_time = time.time() - start_time
            total_time += process_time
            
            # 统计结果
            if true_label:
                if result == true_label:
                    correct_count += 1
                    print("✅ 识别正确")
                else:
                    error_count += 1
                    print(f"❌ 识别错误 - 真实标签: {true_label}")
            else:
                if result == 'background':
                    background_count += 1
                elif result == 'possible_dog':
                    possible_dog_count += 1
        
        # 计算准确率（基于有明确标签的文件）
        evaluable_files = correct_count + error_count
        accuracy = (correct_count / evaluable_files) * 100 if evaluable_files > 0 else 0
        
        # 打印统计结果
        print("\n=================== 测试结果统计 ===================")
        print(f"总测试文件数: {total_files}")
        print(f"明确狗吠声识别正确: {correct_count}")
        print(f"识别错误: {error_count}")
        print(f"识别为背景噪声: {background_count}")
        print(f"识别为可能包含狗叫: {possible_dog_count}")
        print(f"准确率: {accuracy:.2f}% (基于 {evaluable_files} 个可评估文件)")
        print(f"平均处理时间: {(total_time / total_files):.3f}秒/文件")
        print("====================================================")

# 主函数
if __name__ == "__main__":
    print("====================================================")
    print("       🐶 狗吠声纹识别系统 - 量产级鲁棒性测试        ")
    print("====================================================")
    
    try:
        # 创建测试器实例
        tester = RobustDogRecognitionTester()
        
        # 默认测试目录
        test_dir = "./youtube_wav/test"
        
        # 检查测试目录是否存在
        if not os.path.exists(test_dir):
            print(f"❌ 测试目录不存在: {test_dir}")
            # 尝试使用当前目录
            test_dir = "."
            print(f"🔄 尝试使用当前目录: {test_dir}")
        
        # 运行批量测试
        tester.batch_test(test_dir)
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")