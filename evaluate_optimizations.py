#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估优化后的狗吠声识别模型性能
"""

import os
import numpy as np
import librosa
import time
import glob
from enhanced_dog_voice_recognition import DogVoiceModel, Config, preprocess_audio

# 配置
cfg = Config()

# 默认测试目录
default_test_dir = "./youtube_wav/test"

def load_model(model_path="dog_voice_model.pkl"):
    """加载已训练的模型"""
    model = DogVoiceModel()
    try:
        model.load_model(model_path)
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # 如果模型加载失败，创建一个简单的模拟模型用于测试
        print("🔄 创建模拟模型用于测试...")
        
        # 模拟训练数据
        enroll_data = {}
        dog_ids = ['dog1', 'dog2', 'dog3', 'dog4', 'dog5']
        
        for dog_id in dog_ids:
            # 生成100个随机特征向量，每个向量长度为20（MFCC特征数量）
            features = np.random.rand(100, cfg.N_MFCC * 3)  # 假设特征维度为MFCC数量*3
            enroll_data[dog_id] = features
        
        # 训练模型
        model.train_ubm(np.vstack(list(enroll_data.values())))
        model.train_dog_models(enroll_data)
        
        return model

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

def test_model_performance():
    """测试模型性能"""
    # 加载模型
    print("[INFO] 加载模型...")
    model = load_model()
    
    # 测试处理音频的能力
    print("\n[INFO] 开始测试音频处理能力...")
    
    # 测试不同长度的音频
    for duration in [0.2, 0.4, 0.8, 1.5]:
        # 生成测试音频
        sample_rate = cfg.SR
        num_samples = int(duration * sample_rate)
        audio = np.random.randn(num_samples) * 0.1  # 生成随机噪声
        
        try:
            # 预处理音频
            start_time = time.time()
            processed_audio = preprocess_audio(audio)
            process_time = time.time() - start_time
            
            # 识别
            result, similarities = model.recognize(audio)
            
            print(f"\n音频时长: {duration:.2f}秒")
            print(f"处理时间: {process_time*1000:.2f}毫秒")
            print(f"识别结果: {result}")
            print(f"相似度得分: {similarities}")
        except Exception as e:
            print(f"\n音频时长: {duration:.2f}秒")
            print(f"处理失败: {str(e)}")
    
    # 测试特殊情况处理
    print("\n[INFO] 测试特殊情况处理...")
    
    # 空音频
    try:
        result, similarities = model.recognize(None)
        print(f"空音频处理结果: {result}")
    except Exception as e:
        print(f"空音频处理失败: {str(e)}")
    
    # 空数组
    try:
        result, similarities = model.recognize([])
        print(f"空数组处理结果: {result}")
    except Exception as e:
        print(f"空数组处理失败: {str(e)}")
    
    # 非numpy数组
    try:
        result, similarities = model.recognize([0.1, 0.2, 0.3])
        print(f"非numpy数组处理结果: {result}")
    except Exception as e:
        print(f"非numpy数组处理失败: {str(e)}")
    
    # 测试真实文件（如果有）
    print("\n[INFO] 尝试使用真实测试文件...")
    
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
    test_files = get_test_files(test_dir)
    
    if not test_files:
        print("❌ 没有找到测试文件")
    else:
        print(f"✅ 找到 {len(test_files)} 个测试文件")
        
        # 测试前几个文件
        for file_path in test_files[:3]:  # 只测试前3个文件
            file_name = os.path.basename(file_path)
            print(f"\n测试文件: {file_name}")
            
            # 加载音频
            y = load_audio(file_path)
            if y is None:
                continue
            
            # 获取音频时长
            duration = librosa.get_duration(y=y, sr=cfg.SR)
            print(f"⏱️ 音频时长: {duration:.2f}秒")
            
            # 识别
            start_time = time.time()
            try:
                result, similarities = model.recognize(y)
                process_time = time.time() - start_time
                
                print(f"⚡ 处理时间: {process_time*1000:.2f}毫秒")
                print(f"📝 识别结果: {result}")
                if similarities:
                    print("📊 相似度分数:", {k: f"{v:.3f}" for k, v in similarities.items()})
            except Exception as e:
                print(f"❌ 识别失败: {str(e)}")

if __name__ == "__main__":
    test_model_performance()