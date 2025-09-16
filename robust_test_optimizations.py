#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估优化后的狗吠声识别系统在真实音频上的性能
"""

import os
import numpy as np
import librosa
import time
import glob
from enhanced_dog_voice_recognition import DogVoiceModel

# 默认测试目录
default_test_dir = "./youtube_wav/test"

def load_audio(file_path, sr=16000):
    """加载音频文件"""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y
    except Exception as e:
        print(f"❌ 音频加载失败 ({file_path}): {e}")
        return None

def get_test_files(test_dir):
    """获取所有测试文件"""
    search_path = os.path.join(test_dir, "*.WAV")
    test_files = glob.glob(search_path)
    
    if not test_files:
        # 尝试使用小写扩展名
        search_path = os.path.join(test_dir, "*.wav")
        test_files = glob.glob(search_path)
    
    return test_files

def load_model(model_path="dog_voice_model.pkl"):
    """加载已训练的模型"""
    model = DogVoiceModel()
    try:
        model.load_model(model_path)
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # 如果模型加载失败，创建一个简单的模拟模型
        print("🔄 创建模拟模型用于测试...")
        
        # 为每只狗生成模拟训练数据
        enroll_data = {}
        dog_ids = ['dog1', 'dog2', 'dog3', 'dog4', 'dog5']
        
        for dog_id in dog_ids:
            # 为每只狗生成200个随机特征向量，模拟真实特征维度
            features = np.random.rand(200, 60)  # 假设特征维度是60
            enroll_data[dog_id] = features
        
        # 训练UBM
        all_features = np.vstack(list(enroll_data.values()))
        model.train_ubm(all_features)
        
        # 训练每只狗的模型
        model.train_dog_models(enroll_data)
        
        return model

def test_model_performance(test_files, model):
    """测试模型性能"""
    # 统计结果
    total_files = len(test_files)
    correct_count = 0
    error_count = 0
    background_count = 0
    possible_dog_count = 0
    total_time = 0
    
    # 记录每只狗的识别情况
    dog_results = {}
    for dog_id in model.dog_gmms.keys():
        dog_results[dog_id] = {'correct': 0, 'total': 0}
    
    print("\n===== 开始测试 =====")
    print("文件名 | 真实标签 | 识别结果 | 相似度得分 | 是否正确")
    print("-" * 80)
    
    for file_path in test_files:
        # 尝试从文件名提取真实标签
        file_name = os.path.basename(file_path)
        true_label = None
        for dog_id in model.dog_gmms.keys():
            if dog_id in file_name.lower():
                true_label = dog_id
                break
        
        # 加载音频
        y = load_audio(file_path)
        if y is None:
            print(f"{file_name} | {'N/A'} | {'加载失败'} | {'N/A'} | ✗")
            error_count += 1
            continue
        
        # 识别
        start_time = time.time()
        try:
            result, similarities = model.recognize(y)
            process_time = time.time() - start_time
            total_time += process_time
            
            # 判断是否正确
            is_correct = False
            if true_label and result == true_label:
                is_correct = True
                correct_count += 1
                # 更新每只狗的正确数量
                if true_label in dog_results:
                    dog_results[true_label]['correct'] += 1
            elif not true_label and result == 'background':
                is_correct = True
                correct_count += 1
            else:
                error_count += 1
            
            # 更新每只狗的测试数量
            if true_label in dog_results:
                dog_results[true_label]['total'] += 1
            elif not true_label:
                # 背景噪声文件
                background_count += 1
            
            # 统计结果类型
            if result == 'background':
                background_count += 1
            elif result == 'possible_dog':
                possible_dog_count += 1
            
            # 格式化相似度得分
            if similarities:
                sim_str = ", ".join([f"{dog}: {sim:.4f}" for dog, sim in similarities.items()])
            else:
                sim_str = "N/A"
            
            print(f"{file_name} | {true_label or 'N/A'} | {result} | {sim_str} | {'✓' if is_correct else '✗'} (处理时间: {process_time*1000:.2f}ms)")
        except Exception as e:
            error_count += 1
            print(f"{file_name} | {true_label or 'N/A'} | 错误 | {str(e)} | ✗")
    
    # 计算准确率（基于有明确标签的文件）
    evaluable_files = correct_count + error_count
    accuracy = (correct_count / evaluable_files) * 100 if evaluable_files > 0 else 0
    
    print("-" * 80)
    print(f"总体准确率: {accuracy:.2f}% ({correct_count}/{evaluable_files})")
    print(f"识别为背景噪声: {background_count}")
    print(f"识别为可能包含狗叫: {possible_dog_count}")
    print(f"平均处理时间: {(total_time / total_files)*1000:.2f}毫秒/文件")
    
    # 打印每只狗的识别准确率
    print("\n每只狗的识别准确率:")
    for dog_id, stats in dog_results.items():
        if stats['total'] > 0:
            dog_accuracy = (stats['correct'] / stats['total']) * 100
            print(f"{dog_id}: {dog_accuracy:.2f}% ({stats['correct']}/{stats['total']})")
        else:
            print(f"{dog_id}: 无测试数据")
    
    return accuracy

def analyze_short_audio_performance(test_files, model):
    """分析短音频的处理性能"""
    print("\n===== 短音频处理性能分析 =====")
    
    for file_path in test_files:
        # 加载音频
        y = load_audio(file_path)
        if y is None:
            continue
        
        # 获取音频时长
        audio_duration = librosa.get_duration(y=y, sr=16000)
        
        # 识别短音频（小于1秒）
        if audio_duration < 1.0:
            try:
                result, similarities = model.recognize(y)
                
                # 计算特征帧数（近似值）
                frame_length = int(0.025 * 16000)  # 25ms帧长
                hop_length = int(0.01 * 16000)     # 10ms帧移
                num_frames = int((len(y) - frame_length) / hop_length) + 1
                
                file_name = os.path.basename(file_path)
                print(f"{file_name} (时长: {audio_duration:.2f}秒, 帧数: {num_frames}) | 结果: {result}")
                if similarities:
                    print(f"  相似度得分: {similarities}")
            except Exception as e:
                file_name = os.path.basename(file_path)
                print(f"{file_name} (时长: {audio_duration:.2f}秒) | 错误: {str(e)}")

if __name__ == "__main__":
    print("====================================================")
    print("       🐶 狗吠声纹识别系统优化效果评估               ")
    print("====================================================")
    
    try:
        # 加载模型
        print("[INFO] 加载模型...")
        model = load_model()
        
        # 确定测试目录
        test_dir = default_test_dir
        if not os.path.exists(test_dir):
            print(f"❌ 测试目录不存在: {test_dir}")
            # 尝试使用当前目录
            test_dir = os.path.join(".", "wavFiles")
            if not os.path.exists(test_dir):
                test_dir = "."
            print(f"🔄 尝试使用目录: {test_dir}")
        
        # 获取测试文件
        print(f"[INFO] 搜索测试文件...")
        test_files = get_test_files(test_dir)
        
        if not test_files:
            print(f"❌ 没有找到测试文件")
        else:
            print(f"✅ 找到 {len(test_files)} 个测试文件")
            
            # 运行性能测试
            accuracy = test_model_performance(test_files, model)
            
            # 分析短音频性能
            analyze_short_audio_performance(test_files, model)
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        
    print("\n[INFO] 测试完成")