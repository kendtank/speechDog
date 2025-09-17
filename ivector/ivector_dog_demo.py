import os
import argparse
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from ivector_extractor import IVectorExtractor
import logging

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ivector_dog_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 特征提取优化 ==========
def normalize_features(feats):
    """对特征进行均值方差归一化，提高鲁棒性"""
    mean = np.mean(feats, axis=0)
    std = np.std(feats, axis=0)
    std[std < 1e-10] = 1e-10  # 防止除零错误
    return (feats - mean) / std

def extract_mfcc_improved(wav_path, sr=16000, n_mfcc=20):
    """提取MFCC及其差分特征，丰富特征表示
    返回: [frames, 3*n_mfcc] 维度的特征
    """
    try:
        y, sr = librosa.load(wav_path, sr=sr)
        # 预加重
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=512, hop_length=256, n_mels=40)
        
        # 提取一阶差分(ΔMFCC)和二阶差分(ΔΔMFCC)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # 连接三种特征
        combined = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
        feats = combined.T  # [frames, 3*n_mfcc]
        
        # 特征归一化
        feats = normalize_features(feats)
        return feats
    except Exception as e:
        logger.error(f"处理文件 {wav_path} 时出错: {str(e)}")
        raise

# ========== 数据集构建 ==========
def build_dataset(root_dir):
    """
    root_dir 结构：
    root_dir/
      dog1/*.wav
      dog2/*.wav
    """
    X, y = [], []
    dog_ids = sorted(os.listdir(root_dir))
    success_count = 0
    error_count = 0
    
    for label, dog in enumerate(dog_ids):
        dog_path = os.path.join(root_dir, dog)
        if not os.path.isdir(dog_path):
            logger.info(f"跳过非目录项: {dog_path}")
            continue
        
        dog_files = [f for f in os.listdir(dog_path) if f.lower().endswith(".wav")]
        logger.info(f"处理狗 {dog} 的 {len(dog_files)} 个音频文件")
        
        for fname in dog_files:
            fpath = os.path.join(dog_path, fname)
            try:
                # 使用改进的特征提取函数
                feats = extract_mfcc_improved(fpath)
                
                # 过滤掉太短的特征序列
                if feats.shape[0] > 10:  # 至少需要10帧特征
                    X.append(feats)
                    y.append(label)
                    success_count += 1
                else:
                    logger.warning(f"文件特征帧数不足 (仅{feats.shape[0]}帧): {fpath}")
                    error_count += 1
            except Exception as e:
                logger.warning(f"无法处理文件 {fpath}: {str(e)}")
                error_count += 1
    
    logger.info(f"数据集构建完成: 成功处理 {success_count} 个文件, 失败 {error_count} 个文件")
    logger.info(f"总特征序列数: {len(X)}, 狗类别数: {len(set(y))}")
    
    return X, np.array(y), dog_ids

# ========== 构建单个目标狗模型 ==========
def build_target_model(root_dir, target_dog_id):
    """
    为单个目标狗构建模型
    """
    logger.info(f"为狗 {target_dog_id} 构建模型...")
    
    # 收集目标狗的特征
    target_feats = []
    # 收集其他狗的特征（用于UBM训练）
    other_feats = []
    
    dog_ids = sorted(os.listdir(root_dir))
    
    for dog in dog_ids:
        dog_path = os.path.join(root_dir, dog)
        if not os.path.isdir(dog_path):
            continue
            
        dog_files = [f for f in os.listdir(dog_path) if f.lower().endswith(".wav")]
        
        for fname in dog_files:
            fpath = os.path.join(dog_path, fname)
            try:
                feats = extract_mfcc_improved(fpath)
                
                # 过滤掉太短的特征序列
                if feats.shape[0] > 10:
                    if dog == target_dog_id:
                        target_feats.append(feats)
                    else:
                        other_feats.append(feats)
            except Exception as e:
                logger.warning(f"无法处理文件 {fpath}: {str(e)}")
                continue
    
    if len(target_feats) == 0:
        raise ValueError(f"没有找到目标狗 {target_dog_id} 的有效样本")
    
    logger.info(f"目标狗 {target_dog_id} 样本数: {len(target_feats)}")
    logger.info(f"其他狗样本数: {len(other_feats)}")
    
    # 合并特征用于UBM训练
    if other_feats:
        ubm_feats = target_feats + other_feats
    else:
        ubm_feats = target_feats
    
    return target_feats, ubm_feats

# ========== 单狗测试 ==========
def test_single_dog(test_dir, target_dog_id, ivec_model, target_ivectors, threshold=0.5):
    """
    对单个目标狗进行测试
    """
    logger.info(f"开始单狗测试 - 目标狗: {target_dog_id}")
    
    # 获取测试文件
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".wav")]
    test_files.sort()
    
    logger.info(f"测试文件列表: {test_files}")
    logger.info(f"使用阈值: {threshold}")
    logger.info("识别结果:")
    
    correct = 0
    total = 0
    
    for fname in test_files:
        fpath = os.path.join(test_dir, fname)
        try:
            # 提取特征
            feats = extract_mfcc_improved(fpath)
            
            # 提取i-vector
            test_ivector = ivec_model.transform(feats)
            
            # 计算与目标狗i-vectors的相似度
            similarities = []
            for target_ivec in target_ivectors:
                sim = np.dot(test_ivector, target_ivec)
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            
            # 判断是否为目标狗
            is_target = avg_similarity >= threshold
            pred_label = target_dog_id if is_target else "background"
            
            # 获取真实标签
            true_label = "background"
            if fname.lower().startswith(target_dog_id.lower()):
                true_label = target_dog_id
            elif fname.lower().startswith("dog"):
                true_label = fname.split("_")[0]  # 其他狗的ID
            
            # 判断是否正确
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            total += 1
            
            # 输出结果
            status = "✓" if is_correct else "✗"
            logger.info(f"  {fname}: {status} 识别为[{pred_label}] (相似度={avg_similarity:.4f}) 实际为[{true_label}]")
            
        except Exception as e:
            logger.error(f"处理文件 {fname} 时出错: {str(e)}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"[RESULT] 单狗测试准确率: {accuracy:.4f} ({correct}/{total})")
    return accuracy

# ========== 全狗测试 ==========
def test_all_dogs(test_dir, enroll_dir, n_components=4, tv_dim=10):
    """
    对所有狗进行测试
    """
    logger.info("开始全狗测试...")
    
    # 获取所有狗的ID
    dog_ids = sorted([d for d in os.listdir(enroll_dir) if os.path.isdir(os.path.join(enroll_dir, d))])
    logger.info(f"所有已建模狗: {dog_ids}")
    
    # 获取测试文件
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".wav")]
    test_files.sort()
    
    logger.info(f"测试文件列表: {test_files}")
    
    # 为每只狗构建模型并存储
    dog_models = {}
    
    for dog_id in dog_ids:
        try:
            logger.info(f"构建狗 {dog_id} 的模型...")
            target_feats, ubm_feats = build_target_model(enroll_dir, dog_id)
            
            # 训练i-vector提取器
            feature_dim = 60  # 默认MFCC特征维度
            if len(ubm_feats) > 0 and len(ubm_feats[0]) > 0:
                feature_dim = ubm_feats[0].shape[1]
            
            adaptive_tv_dim = min(tv_dim, max(1, feature_dim // 2))
            
            ivec = IVectorExtractor(n_components=n_components, tv_dim=adaptive_tv_dim)
            ivec.fit(ubm_feats)
            
            # 提取目标狗的i-vectors
            target_ivectors = []
            for feats in target_feats:
                ivec_feat = ivec.transform(feats)
                target_ivectors.append(ivec_feat)
            target_ivectors = np.array(target_ivectors)
            
            dog_models[dog_id] = (ivec, target_ivectors)
            logger.info(f"狗 {dog_id} 模型构建完成")
            
        except Exception as e:
            logger.error(f"构建狗 {dog_id} 模型时出错: {str(e)}")
            continue
    
    if not dog_models:
        logger.error("没有成功构建任何狗的模型")
        return
    
    # 测试每个文件
    logger.info("开始测试...")
    results = []
    correct = 0
    total = 0
    
    for fname in test_files:
        logger.info(f"\n{fname}:")
        file_scores = {}
        
        # 获取真实标签
        true_label = "background"
        if fname.lower().startswith("dog"):
            true_label = fname.split("_")[0]
        
        try:
            # 对每只狗计算相似度
            for dog_id, (ivec_model, target_ivectors) in dog_models.items():
                try:
                    fpath = os.path.join(test_dir, fname)
                    feats = extract_mfcc_improved(fpath)
                    test_ivector = ivec_model.transform(feats)
                    
                    # 计算相似度
                    similarities = []
                    for target_ivec in target_ivectors:
                        sim = np.dot(test_ivector, target_ivec)
                        similarities.append(sim)
                    
                    avg_similarity = np.mean(similarities)
                    file_scores[dog_id] = avg_similarity
                    logger.info(f"  对狗 {dog_id} 的相似度 = {avg_similarity:.4f}")
                    
                except Exception as e:
                    logger.warning(f"计算狗 {dog_id} 相似度时出错: {str(e)}")
                    file_scores[dog_id] = 0.0
                    continue
            
            # 选择最高相似度的狗
            if file_scores:
                best_dog = max(file_scores, key=file_scores.get)
                best_score = file_scores[best_dog]
                pred_label = best_dog if best_score >= 0.5 else "background"
                
                is_correct = (pred_label == true_label)
                if is_correct:
                    correct += 1
                total += 1
                
                status = "✓" if is_correct else "✗"
                logger.info(f"  {status} 预测为: {pred_label} (最高相似度={best_score:.4f}) 实际为: {true_label}")
                results.append((fname, pred_label, true_label, best_score, file_scores))
                
        except Exception as e:
            logger.error(f"处理文件 {fname} 时出错: {str(e)}")
            continue
    
    # 输出汇总结果
    logger.info(f"\n[INFO] 所有测试完成")
    logger.info(f"\n[INFO] 汇总结果:")
    for fname, pred, true_label, score, all_scores in results:
        status = "✓" if pred == true_label else "✗"
        logger.info(f"  {fname}: {status} 预测={pred} 实际={true_label} 最高相似度={score:.4f}")
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"\n[RESULT] 总体准确率: {accuracy:.4f} ({correct}/{total})")
    return accuracy

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def adaptive_n_components(data_size, feature_dim):
    """根据数据量和特征维度自适应调整高斯分量数"""
    # 经验公式，确保高斯分量数在合理范围内
    return min(16, max(2, int(np.sqrt(data_size / 10))))

def calculate_eer(genuine_scores, impostor_scores):
    """计算等错误率(EER)作为更精确的评估指标"""
    # 计算FAR和FRR
    fpr, tpr, _ = roc_curve([1]*len(genuine_scores) + [0]*len(impostor_scores), 
                            np.concatenate([genuine_scores, impostor_scores]))
    frr = 1 - tpr
    
    # 找到FAR和FRR最接近的点
    eer_threshold = None
    min_diff = float('inf')
    for i in range(len(fpr)):
        diff = abs(fpr[i] - frr[i])
        if diff < min_diff:
            min_diff = diff
            eer_threshold = i
    
    eer = (fpr[eer_threshold] + frr[eer_threshold]) / 2
    return eer, fpr, tpr

def visualize_results(genuine_scores, impostor_scores, output_dir="./analysis_out"):
    """可视化识别结果，包括相似度分布和ROC曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 相似度分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.5, label='同一狗', color='green')
    plt.hist(impostor_scores, bins=50, alpha=0.5, label='不同狗', color='red')
    plt.legend(loc='upper right')
    plt.xlabel('余弦相似度')
    plt.ylabel('频率')
    plt.title('狗声纹识别相似度分布')
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    plt.close()
    
    # 2. ROC曲线
    eer, fpr, tpr = calculate_eer(genuine_scores, impostor_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.plot([eer], [1-eer], 'ro', markersize=8, label=f'EER = {eer:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('错误接受率 (FAR)')
    plt.ylabel('正确接受率 (TPR)')
    plt.title('狗声纹识别ROC曲线')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    return eer, roc_auc

# ========== 主流程 ==========
def main(data_dir, test_dir, target_dog=None, mode="all", n_components=4, tv_dim=100):
    logger.info("开始狗声纹识别实验")
    logger.info(f"注册数据目录: {data_dir}")
    logger.info(f"测试数据目录: {test_dir}")
    logger.info(f"运行模式: {mode}")
    
    if mode == "single" and target_dog:
        # 单狗测试模式
        logger.info(f"目标狗: {target_dog}")
        
        try:
            # 构建目标狗模型
            target_feats, ubm_feats = build_target_model(data_dir, target_dog)
            
            # 训练i-vector提取器
            logger.info(f"训练i-vector提取器 (高斯分量数: {n_components}, i-vector维度: {tv_dim})...")
            feature_dim = 60  # 默认MFCC特征维度
            if len(ubm_feats) > 0 and len(ubm_feats[0]) > 0:
                feature_dim = ubm_feats[0].shape[1]
            
            adaptive_tv_dim = min(tv_dim, max(1, feature_dim // 2))
            
            ivec = IVectorExtractor(n_components=n_components, tv_dim=adaptive_tv_dim)
            ivec.fit(ubm_feats)
            
            # 提取目标狗的i-vectors
            target_ivectors = []
            for feats in target_feats:
                ivec_feat = ivec.transform(feats)
                target_ivectors.append(ivec_feat)
            target_ivectors = np.array(target_ivectors)
            
            # 执行单狗测试
            test_single_dog(test_dir, target_dog, ivec, target_ivectors)
            
        except Exception as e:
            logger.error(f"单狗测试过程中出错: {str(e)}")
            raise
    else:
        # 全狗测试模式
        try:
            test_all_dogs(test_dir, data_dir, n_components, tv_dim)
        except Exception as e:
            logger.error(f"全狗测试过程中出错: {str(e)}")
            raise
    
    logger.info("实验完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./youtube_wav/brakng_dog_datasets", required=True, help="注册数据集目录 (dog1/, dog2/...)")
    parser.add_argument("--test_dir", type=str, default="./youtube_wav/test", help="测试数据集目录")
    parser.add_argument("--target_dog", type=str, help="目标狗ID (用于单狗测试)")
    parser.add_argument("--mode", type=str, choices=["single", "all"], default="all", help="测试模式: single(单狗) 或 all(全狗)")
    parser.add_argument("--n_components", type=int, default=4, help="GMM 高斯数")
    parser.add_argument("--tv_dim", type=int, default=10, help="i-vector 维度，不能超过特征维度20")
    args = parser.parse_args()

    main(args.data_dir, args.test_dir, args.target_dog, args.mode, args.n_components, args.tv_dim)


"""
使用示例:

# 全狗测试模式
python ivector/ivector_dog_demo.py --data_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test

# 单狗测试模式
python ivector/ivector_dog_demo.py --data_dir ./youtube_wav/brakng_dog_datasets --test_dir ./youtube_wav/test --target_dog dog1 --mode single
"""