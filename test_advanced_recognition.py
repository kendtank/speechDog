#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试高级狗吠声纹识别算法的性能
"""

import os
import numpy as np
import librosa
import pickle
import time
from tabulate import tabulate  # 用于美化表格输出
import matplotlib.pyplot as plt
from collections import defaultdict

# 导入新的高级识别模型
from advanced_dog_voice_recognition import AdvancedDogVoiceModel, preprocess_audio, extract_enhanced_features

# ================== 配置 ==================
class TestConfig:
    def __init__(self):
        # 数据集路径
        self.ENROLL_DIR = "./youtube_wav/brakng_dog_datasets"  # 注册数据目录
        self.TEST_DIR = "./youtube_wav/test"      # 测试数据目录
        self.MODEL_PATH = "./advanced_dog_voice_model.pkl"  # 模型保存路径
        
        # 测试参数
        self.RETRAIN_MODEL = True  # 是否重新训练模型
        self.SAVE_RESULTS = True   # 是否保存测试结果
        self.RESULTS_PATH = "./test_results.csv"  # 测试结果保存路径
        
        # 可视化参数
        self.PLOT_RESULTS = True   # 是否绘制结果
        self.PLOT_PATH = "./recognition_results.png"  # 结果图保存路径

# 全局配置实例
cfg = TestConfig()

# ================== 数据加载函数 ==================

def load_enroll_data(enroll_dir):
    """加载注册数据"""
    enroll_data = defaultdict(list)
    all_features = []
    
    print(f"[INFO] 从 {enroll_dir} 加载注册数据...")
    
    # 遍历每个狗的目录
    for dog_id in os.listdir(enroll_dir):
        dog_dir = os.path.join(enroll_dir, dog_id)
        
        if not os.path.isdir(dog_dir):
            continue
        
        # 收集这只狗的所有音频文件
        audio_files = []
        for file in os.listdir(dog_dir):
            if file.lower().endswith((".wav", ".wave", ".mp3", ".ogg")):
                file_path = os.path.join(dog_dir, file)
                audio_files.append(file_path)
        
        if not audio_files:
            print(f"[WARN] 狗 {dog_id} 的目录中没有找到音频文件")
            continue
        
        print(f"[INFO] 找到狗 {dog_id} 的 {len(audio_files)} 个注册音频文件")
        
        # 提取每个音频文件的特征
        dog_features = []
        for file_path in audio_files:
            try:
                # 加载和预处理音频
                y, _ = librosa.load(file_path, sr=16000)
                y = preprocess_audio(y)
                
                # 提取特征
                features = extract_enhanced_features(y)
                dog_features.append(features)
                all_features.append(features)
                
            except Exception as e:
                print(f"[ERROR] 处理文件 {file_path} 时出错: {str(e)}")
        
        # 合并这只狗的所有特征
        if dog_features:
            enroll_data[dog_id] = np.vstack(dog_features)
    
    if not enroll_data:
        print("[ERROR] 没有找到有效的注册数据")
        return None, None
    
    # 合并所有特征用于训练UBM
    if all_features:
        all_features_combined = np.vstack(all_features)
    else:
        all_features_combined = None
    
    return enroll_data, all_features_combined


def load_test_data(test_dir):
    """加载测试数据"""
    test_data = []
    
    print(f"[INFO] 从 {test_dir} 加载测试数据...")
    
    # 遍历测试目录
    for file in os.listdir(test_dir):
        if not file.lower().endswith((".wav", ".wave", ".mp3", ".ogg")):
            continue
        
        file_path = os.path.join(test_dir, file)
        
        # 从文件名中提取真实标签（假设文件名格式为 dogX_test_XX.wav）
        file_name = os.path.basename(file)
        true_label = "unknown"
        
        # 尝试从文件名提取狗的ID
        for dog_id in ["dog1", "dog2", "dog3", "dog4", "dog5"]:
            if dog_id in file_name.lower():
                true_label = dog_id
                break
        
        # 特殊处理背景噪声文件
        if "bad" in file_name.lower():
            true_label = "background"
        
        test_data.append({
            'file_path': file_path,
            'file_name': file_name,
            'true_label': true_label
        })
    
    if not test_data:
        print("[ERROR] 没有找到有效的测试数据")
        return None
    
    print(f"[INFO] 找到 {len(test_data)} 个测试音频文件")
    return test_data

# ================== 模型训练和评估 ==================

def train_model(enroll_data, all_features):
    """训练高级狗吠声纹识别模型"""
    model = AdvancedDogVoiceModel()
    
    # 训练UBM
    if all_features is not None and len(all_features) > 0:
        model.train_ubm(all_features)
    else:
        print("[WARN] 没有足够的特征来训练UBM，将使用简化的初始化方式")
    
    # 训练每只狗的模型
    model.train_dog_models(enroll_data)
    
    # 保存模型
    model.save_model(cfg.MODEL_PATH)
    print(f"[INFO] 模型已保存到 {cfg.MODEL_PATH}")
    
    return model


def evaluate_model(model, test_data):
    """评估模型性能"""
    print("[INFO] 开始评估模型性能...")
    
    results = []
    dog_correct = defaultdict(int)
    dog_total = defaultdict(int)
    total_correct = 0
    total_background_correct = 0
    total_background = 0
    total_processing_time = 0
    
    # 保存详细的测试结果表格数据
    table_data = []
    
    for sample in test_data:
        file_path = sample['file_path']
        file_name = sample['file_name']
        true_label = sample['true_label']
        
        try:
            # 加载音频
            y, _ = librosa.load(file_path, sr=16000)
            
            # 记录处理时间
            start_time = time.time()
            
            # 识别
            result, similarities = model.recognize(y)
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # 判断是否正确识别
            is_correct = False
            if true_label == "background":
                is_correct = result == "background" or result == "possible_dog"
                total_background += 1
                if is_correct:
                    total_background_correct += 1
            else:
                is_correct = result == true_label
                
            # 更新统计信息
            dog_total[true_label] += 1
            if is_correct:
                dog_correct[true_label] += 1
                total_correct += 1
            
            # 获取最高相似度得分
            max_similarity = max(similarities.values()) if similarities else 0
            
            # 保存结果
            results.append({
                'file_name': file_name,
                'true_label': true_label,
                'predicted_label': result,
                'is_correct': is_correct,
                'similarities': similarities,
                'processing_time': processing_time,
                'max_similarity': max_similarity
            })
            
            # 添加到表格数据
            correct_mark = "✓" if is_correct else "✗"
            table_data.append([
                file_name,
                true_label,
                result,
                f"{max_similarity:.4f}",
                correct_mark
            ])
            
        except Exception as e:
            print(f"[ERROR] 处理文件 {file_name} 时出错: {str(e)}")
            results.append({
                'file_name': file_name,
                'true_label': true_label,
                'predicted_label': 'error',
                'is_correct': False,
                'similarities': {},
                'processing_time': 0,
                'max_similarity': 0
            })
            
            table_data.append([
                file_name,
                true_label,
                'error',
                '0.0000',
                "✗"
            ])
    
    # 计算总体准确率
    total_samples = len(test_data)
    if total_samples > 0:
        overall_accuracy = (total_correct / total_samples) * 100
    else:
        overall_accuracy = 0
    
    # 计算背景检测准确率
    if total_background > 0:
        background_accuracy = (total_background_correct / total_background) * 100
    else:
        background_accuracy = 0
    
    # 计算每只狗的准确率
    dog_accuracies = {}
    for dog_id in sorted(dog_total.keys()):
        if dog_total[dog_id] > 0:
            dog_accuracies[dog_id] = (dog_correct[dog_id] / dog_total[dog_id]) * 100
        else:
            dog_accuracies[dog_id] = 0
    
    # 计算平均处理时间
    avg_processing_time = total_processing_time / total_samples if total_samples > 0 else 0
    
    # 打印详细的测试结果表格
    print("\n===== 详细测试结果 =====")
    headers = ["文件名", "真实标签", "识别结果", "最大相似度", "正确性"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 打印性能统计
    print("\n===== 性能统计 =====")
    print(f"总体准确率: {overall_accuracy:.2f}%")
    print(f"背景检测准确率: {background_accuracy:.2f}%")
    print("每只狗的识别准确率:")
    for dog_id, accuracy in dog_accuracies.items():
        if dog_id != "background":
            print(f"  {dog_id}: {accuracy:.2f}% ({dog_correct[dog_id]}/{dog_total[dog_id]})")
    print(f"平均处理时间: {avg_processing_time*1000:.2f} ms")
    
    return results, overall_accuracy, dog_accuracies, avg_processing_time

# ================== 结果可视化 ==================

def plot_results(results, save_path=None):
    """绘制识别结果可视化图表"""
    if not results:
        return
    
    # 准备数据
    dog_ids = sorted(set(r['true_label'] for r in results if r['true_label'] != 'background'))
    
    # 计算准确率数据
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    for r in results:
        total_counts[r['true_label']] += 1
        if r['is_correct']:
            correct_counts[r['true_label']] += 1
    
    # 计算准确率
    accuracies = {label: (correct_counts[label] / total_counts[label] * 100)
                 for label in total_counts}
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准确率条形图
    labels = list(accuracies.keys())
    values = list(accuracies.values())
    ax1.bar(labels, values, color='skyblue')
    ax1.set_xlabel('狗ID / 类别')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('每类识别准确率')
    ax1.set_ylim(0, 100)
    
    # 为每个条形添加数值标签
    for i, v in enumerate(values):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 相似度分布箱线图
    similarity_data = []
    box_labels = []
    
    for dog_id in dog_ids:
        dog_similarities = [r['max_similarity'] for r in results 
                           if r['true_label'] == dog_id and r['predicted_label'] == dog_id]
        if dog_similarities:
            similarity_data.append(dog_similarities)
            box_labels.append(f'{dog_id} (正确)')
        
        impostor_similarities = [r['max_similarity'] for r in results 
                               if r['true_label'] != dog_id and r['predicted_label'] == dog_id]
        if impostor_similarities:
            similarity_data.append(impostor_similarities)
            box_labels.append(f'{dog_id} (错误)')
    
    if similarity_data:
        ax2.boxplot(similarity_data, labels=box_labels, notch=True)
        ax2.set_xlabel('类别')
        ax2.set_ylabel('相似度得分')
        ax2.set_title('相似度得分分布')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] 结果图已保存到 {save_path}")
    
    plt.show()

# ================== 保存结果 ==================

def save_results(results, save_path):
    """保存测试结果到CSV文件"""
    import csv
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['文件名', '真实标签', '识别结果', '是否正确', '最大相似度', '处理时间(ms)'])
        
        # 写入数据
        for r in results:
            writer.writerow([
                r['file_name'],
                r['true_label'],
                r['predicted_label'],
                '是' if r['is_correct'] else '否',
                f"{r['max_similarity']:.4f}",
                f"{r['processing_time']*1000:.2f}"
            ])
    
    print(f"[INFO] 测试结果已保存到 {save_path}")

# ================== 主函数 ==================

def main():
    print("===== 高级狗吠声纹识别系统测试 ======")
    
    # 加载测试数据
    test_data = load_test_data(cfg.TEST_DIR)
    if not test_data:
        print("[ERROR] 无法加载测试数据，程序退出")
        return
    
    # 加载或训练模型
    if cfg.RETRAIN_MODEL or not os.path.exists(cfg.MODEL_PATH):
        print("[INFO] 正在训练新模型...")
        
        # 加载注册数据
        enroll_data, all_features = load_enroll_data(cfg.ENROLL_DIR)
        if not enroll_data:
            print("[ERROR] 无法加载注册数据，程序退出")
            return
        
        # 训练模型
        model = train_model(enroll_data, all_features)
    else:
        print(f"[INFO] 从 {cfg.MODEL_PATH} 加载已训练的模型...")
        
        # 加载模型
        model = AdvancedDogVoiceModel()
        model.load_model(cfg.MODEL_PATH)
        
        if not model.trained:
            print("[ERROR] 加载的模型未训练，程序退出")
            return
    
    # 评估模型
    results, overall_accuracy, dog_accuracies, avg_processing_time = evaluate_model(model, test_data)
    
    # 保存结果
    if cfg.SAVE_RESULTS:
        save_results(results, cfg.RESULTS_PATH)
    
    # 可视化结果
    if cfg.PLOT_RESULTS:
        plot_results(results, cfg.PLOT_PATH)
    
    print("\n===== 测试完成 =====")
    
if __name__ == "__main__":
    main()