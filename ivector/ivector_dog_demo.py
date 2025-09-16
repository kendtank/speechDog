import os
import argparse
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from ivector_extractor import IVectorExtractor

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
        print(f"[ERROR] 处理文件 {wav_path} 时出错: {str(e)}")
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
            print(f"[INFO] 跳过非目录项: {dog_path}")
            continue
        
        dog_files = [f for f in os.listdir(dog_path) if f.lower().endswith(".wav")]
        print(f"[INFO] 处理狗 {dog} 的 {len(dog_files)} 个音频文件")
        
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
                    print(f"[WARNING] 文件特征帧数不足 (仅{feats.shape[0]}帧): {fpath}")
                    error_count += 1
            except Exception as e:
                print(f"[WARNING] 无法处理文件 {fpath}: {str(e)}")
                error_count += 1
    
    print(f"[INFO] 数据集构建完成: 成功处理 {success_count} 个文件, 失败 {error_count} 个文件")
    print(f"[INFO] 总特征序列数: {len(X)}, 狗类别数: {len(set(y))}")
    
    return X, np.array(y), dog_ids

import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dog_voice_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
def main(data_dir, n_components=4, tv_dim=100):
    logger.info("开始狗声纹识别实验")
    logger.info(f"数据集目录: {data_dir}")
    
    # 加载数据集
    logger.info("加载数据集...")
    X, y, dog_ids = build_dataset(data_dir)
    
    if len(X) == 0:
        logger.error("未能加载到任何有效数据，程序终止")
        return
    
    # 切分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 自适应调整高斯分量数
    feature_dim = X_train[0].shape[1]
    adaptive_components = adaptive_n_components(len(X_train), feature_dim)
    if n_components != adaptive_components:
        logger.info(f"根据数据特征自适应调整高斯分量数: {n_components} → {adaptive_components}")
        n_components = adaptive_components
    
    # 自适应调整i-vector维度
    adaptive_tv_dim = min(tv_dim, feature_dim // 2)  # 确保tv_dim不超过特征维度的一半
    if tv_dim != adaptive_tv_dim:
        logger.info(f"根据特征维度调整i-vector维度: {tv_dim} → {adaptive_tv_dim}")
        tv_dim = adaptive_tv_dim
    
    # 训练i-vector提取器
    logger.info(f"训练i-vector提取器 (高斯分量数: {n_components}, i-vector维度: {tv_dim})...")
    ivec = IVectorExtractor(n_components=n_components, tv_dim=tv_dim)
    ivec.fit(X_train)
    
    # 提取i-vectors（使用批量处理提高效率）
    logger.info("提取i-vectors...")
    iv_train = ivec.batch_transform(X_train)
    iv_test = ivec.batch_transform(X_test)
    
    # 保存训练好的模型
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"ivector_extractor_{n_components}g_{tv_dim}d.pkl")
    ivec.save(model_path)
    
    # 评估
    logger.info("评估模型性能...")
    sims, labels = [], []
    
    # 使用矩阵运算加速相似度计算
    sim_matrix = cosine_similarity(iv_test, iv_train)
    for i in range(len(iv_test)):
        for j in range(len(iv_train)):
            sims.append(sim_matrix[i, j])
            labels.append(int(y_test[i] == y_train[j]))
    
    sims, labels = np.array(sims), np.array(labels)
    pos = sims[labels == 1]
    neg = sims[labels == 0]
    
    # 计算评估指标
    pos_mean = pos.mean()
    neg_mean = neg.mean()
    eer, roc_auc = visualize_results(pos, neg)
    
    # 记录结果
    logger.info(f"评估结果:")
    logger.info(f"  同一狗平均相似度: {pos_mean:.6f}")
    logger.info(f"  不同狗平均相似度: {neg_mean:.6f}")
    logger.info(f"  等错误率(EER): {eer:.6f}")
    logger.info(f"  ROC曲线下面积(AUC): {roc_auc:.6f}")
    logger.info(f"  区分度(同一狗-不同狗相似度): {pos_mean - neg_mean:.6f}")
    
    # 保存结果
    scores_dir = "scores"
    os.makedirs(scores_dir, exist_ok=True)
    
    pos_file = os.path.join(scores_dir, "positive_scores.txt")
    neg_file = os.path.join(scores_dir, "negative_scores.txt")
    
    np.savetxt(pos_file, pos)
    np.savetxt(neg_file, neg)
    
    logger.info(f"相似度分数已保存:")
    logger.info(f"  同一狗分数: {pos_file} (共{len(pos)}个样本)")
    logger.info(f"  不同狗分数: {neg_file} (共{len(neg)}个样本)")
    logger.info("实验完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./youtube_wav/brakng_dog_datasets", required=True, help="数据集目录 (dog1/, dog2/...)")
    parser.add_argument("--n_components", type=int, default=4, help="GMM 高斯数")
    parser.add_argument("--tv_dim", type=int, default=10, help="i-vector 维度，不能超过特征维度20")
    args = parser.parse_args()

    main(args.data_dir, args.n_components, args.tv_dim)


"""
D:/ProgramData/anaconda3/envs/ai/python.exe ivector/ivector_dog_demo.py --data_dir ./youtube_wav/brakng_dog_datasets
"""
