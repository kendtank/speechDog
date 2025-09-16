# -*- coding: utf-8 -*-
"""
使用改进版狗吠声纹识别算法的示例脚本
"""

import os
import numpy as np
import librosa
from enhanced_dog_voice_recognition import DogVoiceModel, Config, preprocess_audio, extract_mfcc_features, augment_audio
import time

# 全局配置
cfg = Config()

# ================== 示例函数 ==================

def prepare_training_data(enroll_dir):
    """准备训练数据"""
    enroll_data = {}
    all_features = []
    
    # 获取所有狗的ID（子目录名）
    dog_ids = [d for d in os.listdir(enroll_dir) if os.path.isdir(os.path.join(enroll_dir, d))]
    
    if not dog_ids:
        print(f"[ERROR] 没有找到训练数据在目录: {enroll_dir}")
        return None, None
    
    print(f"[INFO] 找到 {len(dog_ids)} 只狗的训练数据")
    
    for dog_id in dog_ids:
        dog_dir = os.path.join(enroll_dir, dog_id)
        features_list = []
        file_count = 0
        
        # 获取该狗的所有音频文件
        for file_name in os.listdir(dog_dir):
            if file_name.lower().endswith(('.wav', '.flac')):
                file_path = os.path.join(dog_dir, file_name)
                file_count += 1
                
                try:
                    # 加载和预处理音频
                    print(f"[INFO] 处理文件: {file_path}")
                    y, _ = librosa.load(file_path, sr=cfg.SR)
                    
                    # 数据增强 - 为每个原始样本生成多个增强样本
                    augmented_audios = augment_audio(y)
                    
                    for aug_y in augmented_audios:
                        # 预处理增强后的音频
                        processed_y = preprocess_audio(aug_y)
                        
                        # 提取特征
                        features = extract_mfcc_features(processed_y)
                        features_list.append(features)
                except Exception as e:
                    print(f"[ERROR] 处理文件 {file_path} 时出错: {str(e)}")
        
        if features_list:
            # 合并该狗的所有特征
            dog_features = np.vstack(features_list)
            enroll_data[dog_id] = dog_features
            all_features.append(dog_features)
            print(f"[INFO] 狗 {dog_id}: {file_count} 个原始文件, 生成 {len(features_list)} 个特征样本")
        else:
            print(f"[WARN] 狗 {dog_id} 没有有效的训练样本")
    
    if not all_features:
        print("[ERROR] 没有提取到有效的特征")
        return None, None
    
    # 合并所有狗的特征用于UBM训练
    all_feats_combined = np.vstack(all_features)
    print(f"[INFO] 总共提取特征: {all_feats_combined.shape[0]} 帧, 维度: {all_feats_combined.shape[1]}")
    
    return enroll_data, all_feats_combined


def train_model(enroll_dir, model_path="dog_voice_model.pkl"):
    """训练模型并保存"""
    print("[INFO] 开始训练狗吠声纹识别模型...")
    start_time = time.time()
    
    # 准备训练数据
    enroll_data, all_features = prepare_training_data(enroll_dir)
    
    if enroll_data is None or all_features is None:
        print("[ERROR] 无法训练模型，训练数据为空")
        return False
    
    # 创建并训练模型
    model = DogVoiceModel()
    
    # 训练UBM
    model.train_ubm(all_features)
    
    # 训练每只狗的GMM模型
    model.train_dog_models(enroll_data)
    
    # 保存模型
    model.save_model(model_path)
    
    end_time = time.time()
    print(f"[INFO] 模型训练完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"[INFO] 模型已保存到: {model_path}")
    print(f"[INFO] 成功训练了 {len(model.dog_gmms)} 只狗的模型")
    
    return True


def recognize_dog(audio_path, model_path="dog_voice_model.pkl"):
    """识别狗吠声"""
    # 加载模型
    model = DogVoiceModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {str(e)}")
        return None, None
    
    # 加载并预处理音频
    try:
        y, _ = librosa.load(audio_path, sr=cfg.SR)
    except Exception as e:
        print(f"[ERROR] 加载音频文件失败: {str(e)}")
        return None, None
    
    # 识别
    result, scores = model.recognize(y)
    
    return result, scores


def batch_recognition(test_dir, model_path="dog_voice_model.pkl"):
    """批量识别测试目录中的所有音频文件"""
    print(f"[INFO] 开始批量识别测试目录: {test_dir}")
    
    # 加载模型
    model = DogVoiceModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {str(e)}")
        return
    
    # 获取测试文件列表
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.wav', '.flac'))]
    
    if not test_files:
        print(f"[ERROR] 在目录 {test_dir} 中没有找到音频文件")
        return
    
    print(f"[INFO] 找到 {len(test_files)} 个测试文件")
    
    # 记录正确和错误的预测
    correct_count = 0
    error_count = 0
    possible_dog_count = 0
    background_count = 0
    total_time = 0
    
    # 开始识别
    for file_name in test_files:
        file_path = os.path.join(test_dir, file_name)
        print(f"\n[INFO] 处理文件: {file_name}")
        
        # 记录处理时间
        start_time = time.time()
        
        try:
            # 加载音频
            y, _ = librosa.load(file_path, sr=cfg.SR)
            
            # 识别
            result, scores = model.recognize(y)
            
            # 计算处理时间
            process_time = time.time() - start_time
            total_time += process_time
            
            # 尝试从文件名推断真实标签
            true_label = None
            base_name = file_name.lower()
            for dog_id in model.dog_gmms.keys():
                if dog_id.lower() in base_name:
                    true_label = dog_id
                    break
            
            # 如果文件名中没有狗的ID，假设是未知狗
            if true_label is None:
                true_label = 'unknown'
            
            # 打印识别结果
            print(f"  识别结果: {result}")
            print(f"  真实标签: {true_label}")
            print(f"  各狗得分: { {k: round(v, 4) for k, v in scores.items()} }")
            print(f"  处理时间: {process_time:.3f} 秒")
            
            # 统计准确率
            if result == 'possible_dog':
                # 对于可能包含狗叫的背景噪声，单独统计
                possible_dog_count += 1
                print("  ⚠️ 可能包含狗叫，无法确定具体狗只")
            elif result == 'background':
                background_count += 1
                if true_label == 'unknown':
                    # 背景噪声识别正确
                    correct_count += 1
                    print("  ✅ 正确识别为背景噪声")
                else:
                    # 背景噪声识别错误
                    error_count += 1
                    print("  ❌ 错误识别为背景噪声")
            elif result == true_label:
                # 狗只识别正确
                correct_count += 1
                print("  ✅ 识别正确")
            else:
                # 狗只识别错误
                error_count += 1
                print("  ❌ 识别错误")
                
        except Exception as e:
            error_count += 1
            print(f"  [ERROR] 处理失败: {str(e)}")
    
    # 打印统计信息
    total_files = len(test_files)
    if total_files > 0:
        # 计算准确率时考虑所有可评估的文件（排除possible_dog）
        evaluable_files = total_files - possible_dog_count
        if evaluable_files > 0:
            accuracy = correct_count / evaluable_files * 100
        else:
            accuracy = 0.0
            
        avg_time = total_time / total_files
        
        print(f"\n=== 识别统计 ===")
        print(f"总文件数: {total_files}")
        print(f"明确狗吠声识别正确: {correct_count}")
        print(f"识别错误: {error_count}")
        print(f"可能包含狗叫: {possible_dog_count}")
        print(f"明确背景噪声: {background_count}")
        print(f"准确率: {accuracy:.2f}% (基于 {evaluable_files} 个可评估文件)")
        print(f"平均处理时间: {avg_time:.3f} 秒/文件")

# ================== 主函数 ==================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="狗吠声纹识别示例")
    parser.add_argument('--mode', type=str, choices=['train', 'recognize', 'batch'], 
                        default='batch', help='运行模式')
    parser.add_argument('--enroll_dir', type=str, 
                        default='./youtube_wav/brakng_dog_datasets', help='训练数据目录')
    parser.add_argument('--test_dir', type=str, 
                        default='./youtube_wav/test', help='测试数据目录')
    parser.add_argument('--audio_file', type=str, 
                        help='单文件识别的音频路径')
    parser.add_argument('--model_path', type=str, 
                        default='dog_voice_model.pkl', help='模型保存路径')
    
    args = parser.parse_args()
    
    # 根据模式运行不同的功能
    if args.mode == 'train':
        # 训练模式
        train_model(args.enroll_dir, args.model_path)
        
    elif args.mode == 'recognize':
        # 单文件识别模式
        if not args.audio_file:
            print("[ERROR] 请提供音频文件路径")
            print("使用示例: python example_usage.py --mode recognize --audio_file your_audio.wav")
        else:
            result, scores = recognize_dog(args.audio_file, args.model_path)
            if result is not None:
                print(f"\n识别结果: {result}")
                print(f"各狗得分: { {k: round(v, 4) for k, v in scores.items()} }")
    
    elif args.mode == 'batch':
        # 批量识别模式
        batch_recognition(args.test_dir, args.model_path)

# ================== 运行示例 ==================
# 
# 训练模型:
# D:\ProgramData\anaconda3\envs\ai\python.exe example_usage.py --mode train --enroll_dir ./youtube_wav/brakng_dog_datasets
# 单文件识别:
# python example_usage.py --mode recognize --audio_file ./test_audio.wav
# 
# 批量识别:
# D:\ProgramData\anaconda3\envs\ai\python.exe example_usage.py --mode batch --test_dir ./youtube_wav/test
#