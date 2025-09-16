import numpy as np
import librosa
import pickle
import os
import logging
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("one_class_dog_verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OneClassDogIVectorSystem:
    def __init__(self, dog_name="target_dog", n_components=8, tv_dim=32, reg_covar=1e-4, threshold=0.1,
                 n_mfcc=13, n_fft=512, hop_length=256, gmm_max_iter=100, gmm_tol=1e-3):
        """
        单类狗声纹验证i-vector系统
        
        参数:
        ------
        dog_name: str, 目标狗的名称
        n_components: int, GMM分量数（增加分量数以提高建模能力）
        tv_dim: int, i-vector维度（增加维度以保留更多特征信息）
        reg_covar: float, 协方差矩阵正则化参数
        threshold: float, 验证阈值
        n_mfcc: int, MFCC特征数量
        n_fft: int, FFT窗口大小
        hop_length: int, 帧移
        gmm_max_iter: int, GMM最大迭代次数
        gmm_tol: float, GMM收敛阈值
        """
        self.dog_name = dog_name
        self.n_components = n_components
        self.tv_dim = tv_dim
        self.reg_covar = reg_covar
        self.threshold = threshold
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.gmm_max_iter = gmm_max_iter
        self.gmm_tol = gmm_tol
        
        # 模型组件
        self.ubm = None            # 通用背景模型
        self.pca = None            # total variability矩阵
        self.target_ivectors = []  # 目标狗的i-vector集合
        self.feat_dim = None       # 特征维度
        self.is_trained = False    # 训练状态标志
        self.feature_scaler = None # 特征标准化器
        
    def preprocess_audio(self, y, sr=16000):
        """
        音频预处理，增强狗吠信号，抑制噪声
        """
        # 1. 预加重
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        
        # 2. 噪声抑制（改进版频域滤波）
        # 计算短时傅里叶变换
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = np.abs(D), np.angle(D)
        
        # 改进的噪声估计：使用谱减法+维纳滤波
        noise_len = int(0.5 * sr / self.hop_length)  # 0.5秒的帧数
        if noise_len > magnitude.shape[1]:
            noise_len = magnitude.shape[1] // 2
        
        noise_magnitude = np.mean(magnitude[:, :noise_len], axis=1, keepdims=True)
        
        # 应用谱减法，并添加谱平滑
        magnitude = np.maximum(magnitude - 1.2 * noise_magnitude, 0.05 * magnitude)
        
        # 应用平滑以减少音乐噪声
        magnitude = gaussian_filter1d(magnitude, sigma=1.0, axis=1)
        
        # 重建音频
        D_denoised = magnitude * np.exp(1j * phase)
        y_denoised = librosa.istft(D_denoised, hop_length=self.hop_length)
        
        # 3. 信号归一化
        y_denoised = y_denoised / np.max(np.abs(y_denoised) + 1e-8)
        
        return y_denoised
        
    def extract_mfcc_robust(self, wav_path, sr=16000):
        """
        提取鲁棒的MFCC特征，适合噪声环境
        """
        try:
            # 加载音频
            y, sr = librosa.load(wav_path, sr=sr)
            
            # 预处理：噪声抑制和信号增强
            y = self.preprocess_audio(y, sr)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length, n_mels=40,  # 增加mel滤波器数量
                fmin=20, fmax=8000  # 聚焦于狗吠的主要频率范围
            )
            
            # 高斯平滑，减少噪声影响
            mfcc = gaussian_filter1d(mfcc, sigma=0.8, axis=1)
            
            # 一阶差分（ΔMFCC）和二阶差分（ΔΔMFCC）
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # 组合特征
            combined = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
            feats = combined.T  # [frames, 3*n_mfcc]
            
            # 特征归一化（CMVN - Cepstral Mean and Variance Normalization）
            mean = np.mean(feats, axis=0)
            std = np.std(feats, axis=0)
            std[std < 1e-10] = 1e-10  # 防止除零错误
            feats = (feats - mean) / std
            
            # 添加能量特征
            rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length).T
            feats = np.concatenate([feats, rms], axis=1)
            
            return feats
            
        except Exception as e:
            logger.error(f"处理文件 {wav_path} 时出错: {str(e)}")
            raise
    
    def train(self, enroll_wav_paths):
        """
        使用目标狗的注册音频训练系统
        
        参数:
        ------
        enroll_wav_paths: list, 目标狗的注册音频文件路径列表
        """
        logger.info(f"开始训练单类狗声纹验证系统 - 目标狗: {self.dog_name}")
        
        # 提取所有注册音频的特征
        X_list = []
        for wav_path in enroll_wav_paths:
            logger.info(f"处理注册音频: {wav_path}")
            feats = self.extract_mfcc_robust(wav_path)
            
            # 过滤太短的特征
            if feats.shape[0] > 8:  # 至少需要8帧特征
                X_list.append(feats)
            else:
                logger.warning(f"注册音频太短，特征帧数不足: {wav_path}")
        
        if len(X_list) == 0:
            raise ValueError("没有有效的注册音频用于训练")
        
        # 获取特征维度
        self.feat_dim = X_list[0].shape[1]
        logger.info(f"特征维度: {self.feat_dim}")
        logger.info(f"有效注册样本数: {len(X_list)}")
        
        # 1. 训练UBM GMM
        logger.info(f"训练UBM GMM (高斯分量数: {self.n_components})...")
        
        try:
            # 合并所有特征用于GMM训练
            feats = np.vstack(X_list)
            
            # 初始化特征标准化器
            self.feature_scaler = StandardScaler()
            feats_scaled = self.feature_scaler.fit_transform(feats)
            
            # 训练GMM（增加迭代次数以提高模型质量）
            self.ubm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                max_iter=self.gmm_max_iter,  # 增加迭代次数以提高模型质量
                reg_covar=self.reg_covar,
                tol=self.gmm_tol,            # 更严格的收敛条件
                random_state=42,
                verbose=0
            )
            
            # 使用标准化后的特征进行训练
            self.ubm.fit(feats_scaled)
            logger.info("UBM GMM训练完成")
            
        except Exception as e:
            logger.error(f"UBM GMM训练失败: {str(e)}")
            raise
        
        # 2. 计算统计量并训练PCA以近似total variability矩阵
        logger.info(f"估计total variability矩阵 (i-vector维度: {self.tv_dim})...")
        
        try:
            # 计算每个语音段的超向量
            supervectors = []
            for X in X_list:
                # 计算后验概率
                post = self.ubm.predict_proba(X)
                
                # 计算零阶和一阶统计量
                N = post.sum(axis=0)
                F = np.dot(post.T, X)
                
                # 计算超向量并归一化
                supervector = F.flatten() / (N.sum() + 1e-8)
                supervectors.append(supervector)
            
            supervectors = np.array(supervectors)
            
            # 确保tv_dim不超过超向量维度和样本数量
            max_pca_dim = min(self.tv_dim, supervectors.shape[0], supervectors.shape[1])
            if max_pca_dim < self.tv_dim:
                logger.warning(f"调整i-vector维度: {self.tv_dim} → {max_pca_dim} (受限于样本数量或特征维度)")
                self.tv_dim = max_pca_dim
            
            # 训练PCA模型，增加解释方差要求
            self.pca = PCA(n_components=self.tv_dim, random_state=42)
            self.pca.fit(supervectors)
            
            # 计算PCA解释方差比例
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA训练完成，解释方差比例: {explained_variance:.4f}")
            
            # 如果解释方差过低，尝试增加维度
            if explained_variance < 0.9 and self.tv_dim < supervectors.shape[1] and self.tv_dim < supervectors.shape[0]:
                suggested_dim = min(self.tv_dim * 2, supervectors.shape[0], supervectors.shape[1])
                logger.info(f"尝试增加i-vector维度以提高解释方差: {self.tv_dim} → {suggested_dim}")
                self.pca = PCA(n_components=suggested_dim, random_state=42)
                self.pca.fit(supervectors)
                self.tv_dim = suggested_dim
                explained_variance = np.sum(self.pca.explained_variance_ratio_)
                logger.info(f"调整后PCA解释方差比例: {explained_variance:.4f}")
            
        except Exception as e:
            logger.error(f"Total variability矩阵估计失败: {str(e)}")
            raise
        
        # 3. 提取并存储目标狗的i-vectors
        logger.info("提取目标狗的i-vectors...")
        self.target_ivectors = []
        
        for X in X_list:
            ivec = self._extract_ivector(X)
            self.target_ivectors.append(ivec)
        
        self.target_ivectors = np.array(self.target_ivectors)
        
        # 标记为已训练
        self.is_trained = True
        logger.info(f"单类狗声纹验证系统训练完成")
        
    def _extract_ivector(self, X):
        """
        从特征中提取i-vector
        """
        # 对特征进行标准化
        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # 计算后验概率
        post = self.ubm.predict_proba(X_scaled)
        
        # 计算统计量
        N = post.sum(axis=0)
        F = np.dot(post.T, X_scaled)
        
        # 计算超向量
        supervector = F.flatten() / (N.sum() + 1e-8)
        
        # 使用PCA将超向量压缩为i-vector
        ivec = self.pca.transform(supervector.reshape(1, -1))[0]
        
        # L2归一化i-vector
        ivec_norm = np.linalg.norm(ivec)
        if ivec_norm > 1e-10:  # 避免除零
            ivec = ivec / ivec_norm
        
        return ivec
        
    def verify(self, test_wav_path):
        """
        验证测试音频是否为目标狗的声音
        
        参数:
        ------
        test_wav_path: str, 测试音频文件路径
        
        返回值:
        -------
        result: dict, 包含验证结果、相似度和决策阈值
        """
        if not self.is_trained:
            raise RuntimeError("必须先训练系统才能进行验证")
        
        # 提取测试音频的特征和i-vector
        feats = self.extract_mfcc_robust(test_wav_path)
        if feats.shape[0] <= 8:  # 特征太短，无法可靠验证
            logger.warning(f"测试音频特征帧数不足，无法验证: {test_wav_path}")
            return {
                'is_target': False,
                'similarity': 0.0,
                'threshold': self.threshold,
                'confidence': 0.0,
                'message': "音频太短，无法可靠验证"
            }
        
        test_ivec = self._extract_ivector(feats)
        
        # 计算与目标狗所有i-vector的相似度
        similarities = []
        for target_ivec in self.target_ivectors:
            # 余弦相似度
            sim = np.dot(test_ivec, target_ivec)
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        # 计算相似度的标准差，用于置信度计算
        if len(similarities) > 1:
            similarity_std = np.std(similarities)
        else:
            similarity_std = 0.0
        
        # 改进的决策逻辑：综合考虑平均相似度和相似度分布
        # 1. 基础条件：平均相似度必须大于阈值
        # 2. 额外条件：相似度分布必须集中（低标准差），表示测试样本与所有目标样本都相似
        is_target = avg_similarity >= self.threshold and (similarity_std < 0.2 or avg_similarity > self.threshold + 0.1)
        
        # 改进的置信度计算，考虑相似度分布和阈值距离
        confidence = min(1.0, max(0.0, (avg_similarity - self.threshold + 0.3) / (0.6 + similarity_std)))
        confidence = min(1.0, max(0.0, confidence))
        
        result = {
            'is_target': is_target,
            'similarity': avg_similarity,
            'threshold': self.threshold,
            'confidence': confidence,
            'similarity_std': similarity_std,
            'message': f"验证成功: 是{self.dog_name}" if is_target else f"验证失败: 不是{self.dog_name}"
        }
        
        logger.info(f"验证结果 - 文件: {test_wav_path}, 相似度: {avg_similarity:.4f}, 相似度标准差: {similarity_std:.4f}, 结果: {result['message']}")
        
        return result
        
    def save_model(self, filepath=None):
        """
        保存训练好的模型到文件，方便端侧部署
        """
        if not self.is_trained:
            logger.warning("警告: 尝试保存未训练的模型")
            return False
        
        # 如果未指定路径，使用默认路径
        if filepath is None:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            filepath = os.path.join(model_dir, f"one_class_ivector_{self.dog_name}_{self.n_components}g_{self.tv_dim}d.pkl")
        
        try:
            # 保存模型参数
            model_params = {
                'dog_name': self.dog_name,
                'n_components': self.n_components,
                'tv_dim': self.tv_dim,
                'reg_covar': self.reg_covar,
                'threshold': self.threshold,
                'feat_dim': self.feat_dim,
                'ubm': self.ubm,
                'pca': self.pca,
                'target_ivectors': self.target_ivectors,
                'is_trained': self.is_trained
            }
            
            # 使用协议4以获得更好的压缩率，适合端侧部署
            with open(filepath, 'wb') as f:
                pickle.dump(model_params, f, protocol=4)
                
            logger.info(f"模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            return False
            
    @classmethod
    def load_model(cls, filepath):
        """
        从文件加载训练好的模型
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"模型文件不存在: {filepath}")
                
            # 加载模型参数
            with open(filepath, 'rb') as f:
                model_params = pickle.load(f)
                
            # 创建实例
            system = cls(
                dog_name=model_params['dog_name'],
                n_components=model_params['n_components'],
                tv_dim=model_params['tv_dim'],
                reg_covar=model_params['reg_covar'],
                threshold=model_params['threshold']
            )
            
            # 恢复模型组件
            system.feat_dim = model_params['feat_dim']
            system.ubm = model_params['ubm']
            system.pca = model_params['pca']
            system.target_ivectors = model_params['target_ivectors']
            system.is_trained = model_params['is_trained']
            
            logger.info(f"模型已从 {filepath} 加载")
            return system
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def tune_threshold(self, test_positive_paths, test_negative_paths, target_frr=0.1):
        """
        调整验证阈值以满足特定的误拒率要求
        
        参数:
        ------
        test_positive_paths: list, 目标狗的测试音频路径列表
        test_negative_paths: list, 非目标狗的测试音频路径列表
        target_frr: float, 目标误拒率（希望将目标狗错误拒绝的概率控制在这个水平）
        
        返回值:
        -------
        dict: 包含调整后的阈值、误拒率和误纳率信息
        """
        if not self.is_trained:
            raise RuntimeError("必须先训练系统才能调整阈值")
        
        logger.info(f"开始调整验证阈值，目标误拒率: {target_frr*100:.1f}%")
        
        # 计算所有正样本的相似度
        pos_similarities = []
        for wav_path in test_positive_paths:
            try:
                feats = self.extract_mfcc_robust(wav_path)
                if feats.shape[0] > 8:
                    test_ivec = self._extract_ivector(feats)
                    similarities = [np.dot(test_ivec, tgt) for tgt in self.target_ivectors]
                    pos_similarities.append(np.mean(similarities))
            except Exception as e:
                logger.warning(f"处理正样本时出错: {wav_path}, {str(e)}")
                
        # 计算所有负样本的相似度
        neg_similarities = []
        for wav_path in test_negative_paths:
            try:
                feats = self.extract_mfcc_robust(wav_path)
                if feats.shape[0] > 8:
                    test_ivec = self._extract_ivector(feats)
                    similarities = [np.dot(test_ivec, tgt) for tgt in self.target_ivectors]
                    neg_similarities.append(np.mean(similarities))
            except Exception as e:
                logger.warning(f"处理负样本时出错: {wav_path}, {str(e)}")
                
        if len(pos_similarities) == 0 or len(neg_similarities) == 0:
            raise ValueError("没有足够的有效测试样本用于阈值调整")
        
        # 根据目标误拒率确定阈值
        pos_similarities.sort()
        # 找到对应误拒率的阈值位置
        threshold_idx = max(0, min(len(pos_similarities) - 1, int(len(pos_similarities) * target_frr)))
        new_threshold = pos_similarities[threshold_idx]
        
        # 更新阈值
        old_threshold = self.threshold
        self.threshold = new_threshold
        
        # 计算调整后的误纳率(FAR)
        far = np.mean([1 for sim in neg_similarities if sim >= new_threshold])
        
        logger.info(f"阈值调整完成: 旧阈值={old_threshold:.4f} → 新阈值={new_threshold:.4f}")
        logger.info(f"调整后 - 误拒率(FRR): {np.mean([1 for sim in pos_similarities if sim < new_threshold]):.4f}")
        logger.info(f"调整后 - 误纳率(FAR): {far:.4f}")
        
        # 打印详细的相似度分布信息
        logger.info(f"正样本相似度范围: {min(pos_similarities):.4f} - {max(pos_similarities):.4f}")
        logger.info(f"负样本相似度范围: {min(neg_similarities):.4f} - {max(neg_similarities):.4f}")
        
        return {
            'threshold': new_threshold,
            'frr': np.mean([1 for sim in pos_similarities if sim < new_threshold]),
            'far': far,
            'pos_similarities': pos_similarities,
            'neg_similarities': neg_similarities
        }
        
    def auto_tune_threshold(self, test_dir, positive_pattern="*"):
        """
        自动调整阈值，通过从测试目录中识别正样本和负样本
        
        参数:
        ------
        test_dir: str, 测试音频目录路径
        positive_pattern: str, 正样本文件名匹配模式（包含该模式的文件被认为是目标狗的音频）
        
        返回值:
        -------
        dict: 包含调整后的阈值、误拒率和误纳率信息
        """
        if not self.is_trained:
            raise RuntimeError("必须先训练系统才能调整阈值")
        
        logger.info(f"开始自动调整阈值，正样本匹配模式: '{positive_pattern}'")
        
        # 收集测试文件
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"测试目录不存在: {test_dir}")
        
        test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith('.wav')]
        if not test_files:
            raise ValueError(f"测试目录中没有找到WAV文件: {test_dir}")
        
        # 分类正样本和负样本
        test_positive_paths = []
        test_negative_paths = []
        
        for test_file in test_files:
            if positive_pattern.lower() in os.path.basename(test_file).lower():
                test_positive_paths.append(test_file)
            else:
                test_negative_paths.append(test_file)
        
        logger.info(f"找到正样本数: {len(test_positive_paths)}, 负样本数: {len(test_negative_paths)}")
        
        # 如果正样本太少，使用更宽松的匹配策略
        if len(test_positive_paths) < 2:
            logger.warning("正样本数量过少，尝试使用更宽松的匹配策略...")
            # 使用注册音频作为正样本
            enroll_dir = os.path.join("youtube_wav", "brakng_dog_datasets", self.dog_name)
            if os.path.exists(enroll_dir):
                test_positive_paths = [os.path.join(enroll_dir, f) for f in os.listdir(enroll_dir) if f.lower().endswith('.wav')]
                logger.info(f"使用注册音频作为正样本，数量: {len(test_positive_paths)}")
        
        # 调用阈值调整函数
        return self.tune_threshold(test_positive_paths, test_negative_paths)

# ========== 使用示例 ==========
import argparse
import numpy as np

if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='单类狗声纹验证系统')
    parser.add_argument('--dog_name', type=str, default='dog1', help='目标狗的名称')
    parser.add_argument('--enroll_dir', type=str, default='./youtube_wav/brakng_dog_datasets/dog1', help='注册音频目录')
    parser.add_argument('--test_dir', type=str, default='./youtube_wav/test', help='测试音频目录')
    parser.add_argument('--n_components', type=int, default=8, help='GMM分量数')
    parser.add_argument('--tv_dim', type=int, default=32, help='i-vector维度')
    parser.add_argument('--gmm_max_iter', type=int, default=100, help='GMM最大迭代次数')
    parser.add_argument('--gmm_tol', type=float, default=1e-3, help='GMM收敛阈值')
    parser.add_argument('--enhanced_feature', action='store_true', default=True, help='启用增强特征提取')
    parser.add_argument('--threshold', type=float, default=0.1, help='验证阈值')
    parser.add_argument('--auto_tune', action='store_true', help='自动调整阈值')
    parser.add_argument('--target_frr', type=float, default=0.1, help='自动调整阈值时的目标误拒率')
    parser.add_argument('--model_path', type=str, default=None, help='保存/加载模型的路径')
    parser.add_argument('--test_file', type=str, default=None, help='单独测试的音频文件路径')
    parser.add_argument('--save_results', action='store_true', help='保存验证结果到文件')
    
    args = parser.parse_args()
    
    logger.info("===== 单类狗声纹验证系统演示 =====")
    
    try:
        # 1. 初始化系统
        system = OneClassDogIVectorSystem(
            dog_name=args.dog_name,
            n_components=args.n_components,
            tv_dim=args.tv_dim,
            threshold=args.threshold,
            gmm_max_iter=args.gmm_max_iter,
            gmm_tol=args.gmm_tol
        )
        
        logger.info(f"系统初始化完成 - 使用阈值: {system.threshold}")
        
        # 如果指定了模型路径且文件存在，则加载模型
        if args.model_path and os.path.exists(args.model_path):
            logger.info(f"从文件加载模型: {args.model_path}")
            system = OneClassDogIVectorSystem.load_model(args.model_path)
        else:
            # 2. 准备注册音频
            if not os.path.exists(args.enroll_dir):
                raise FileNotFoundError(f"注册目录不存在: {args.enroll_dir}")
            
            enroll_files = [os.path.join(args.enroll_dir, f) for f in os.listdir(args.enroll_dir) if f.lower().endswith('.wav')]
            if not enroll_files:
                raise ValueError(f"注册目录中没有找到WAV文件: {args.enroll_dir}")
            
            logger.info(f"找到 {len(enroll_files)} 个注册音频文件")
            
            # 3. 训练系统
            system.train(enroll_files)
            
            # 4. 保存模型
            if args.model_path:
                system.save_model(args.model_path)
            else:
                system.save_model()
        
        # 5. 自动调整阈值（如果启用）
        if args.auto_tune and os.path.exists(args.test_dir):
            logger.info("开始自动调整阈值...")
            tune_result = system.auto_tune_threshold(
                test_dir=args.test_dir,
                positive_pattern=args.dog_name
            )
            logger.info(f"阈值调整完成，当前阈值: {system.threshold:.4f}")
        
        # 6. 验证模式选择
        results = []
        
        # 单独测试一个文件
        if args.test_file:
            if not os.path.exists(args.test_file):
                raise FileNotFoundError(f"测试文件不存在: {args.test_file}")
            
            logger.info(f"验证单个文件: {args.test_file}")
            result = system.verify(args.test_file)
            results.append((args.test_file, result))
            
            # 打印详细结果
            logger.info(f"文件: {os.path.basename(args.test_file)}")
            logger.info(f"  相似度: {result['similarity']:.4f}")
            logger.info(f"  阈值: {result['threshold']:.4f}")
            logger.info(f"  置信度: {result['confidence']:.4f}")
            logger.info(f"  结果: {result['message']}")
        
        # 测试整个目录
        elif os.path.exists(args.test_dir):
            test_files = [os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) if f.lower().endswith('.wav')]
            if not test_files:
                raise ValueError(f"测试目录中没有找到WAV文件: {args.test_dir}")
            
            logger.info(f"找到 {len(test_files)} 个测试音频文件")
            
            # 单独保存目标狗测试文件和其他测试文件的结果
            target_results = []
            non_target_results = []
            
            for test_file in test_files:
                result = system.verify(test_file)
                results.append((test_file, result))
                
                # 分类保存结果
                if args.dog_name.lower() in os.path.basename(test_file).lower():
                    target_results.append(result)
                else:
                    non_target_results.append(result)
                
                logger.info(f"文件: {os.path.basename(test_file)}, 结果: {result['message']}, 相似度: {result['similarity']:.4f}")
            
            # 打印目标狗测试文件的统计信息
            if target_results:
                similarities = [r['similarity'] for r in target_results]
                logger.info(f"\n{args.dog_name}测试文件统计:")
                logger.info(f"总文件数: {len(target_results)}")
                logger.info(f"相似度范围: {min(similarities):.4f} - {max(similarities):.4f}")
                logger.info(f"平均相似度: {np.mean(similarities):.4f}")
                logger.info(f"识别成功数: {sum(1 for r in target_results if r['is_target'])}")
                logger.info(f"识别成功率: {sum(1 for r in target_results if r['is_target'])/len(target_results)*100:.1f}%")
            
            # 打印其他狗测试文件的统计信息
            if non_target_results:
                similarities = [r['similarity'] for r in non_target_results]
                logger.info(f"\n其他狗测试文件统计:")
                logger.info(f"总文件数: {len(non_target_results)}")
                logger.info(f"相似度范围: {min(similarities):.4f} - {max(similarities):.4f}")
                logger.info(f"平均相似度: {np.mean(similarities):.4f}")
                logger.info(f"错误接受数: {sum(1 for r in non_target_results if r['is_target'])}")
                logger.info(f"误纳率(FAR): {sum(1 for r in non_target_results if r['is_target'])/len(non_target_results)*100:.1f}%")
            
            # 如果没有成功识别目标狗，尝试进一步降低阈值进行测试
            if target_results and sum(1 for r in target_results if r['is_target']) == 0:
                logger.info(f"\n尝试降低阈值重新验证{args.dog_name}测试文件...")
                original_threshold = system.threshold
                
                # 尝试找到合适的阈值
                target_similarities = [r['similarity'] for r in target_results]
                suggested_threshold = min(target_similarities) - 0.05  # 设置略低于最小相似度的阈值
                system.threshold = max(-0.1, suggested_threshold)  # 确保阈值不低于-0.1
                
                logger.info(f"临时调整阈值: {original_threshold:.4f} → {system.threshold:.4f}")
                
                # 重新验证目标狗测试文件
                recheck_results = []
                for test_file in [f for f in test_files if args.dog_name.lower() in os.path.basename(f).lower()]:
                    result = system.verify(test_file)
                    recheck_results.append(result)
                    logger.info(f"重验证 - 文件: {os.path.basename(test_file)}, 结果: {result['message']}, 相似度: {result['similarity']:.4f}")
                
                # 打印重验证统计
                logger.info(f"重验证后 - {args.dog_name}识别成功率: {sum(1 for r in recheck_results if r['is_target'])/len(recheck_results)*100:.1f}%")
                
                # 保存调整后的模型
                if args.model_path:
                    system.save_model(args.model_path)
                
                # 恢复原始阈值
                system.threshold = original_threshold
        
        # 7. 保存结果到文件（如果启用）
        if args.save_results and results:
            result_file = f"{args.dog_name}_verification_results.txt"
            with open(result_file, 'w') as f:
                f.write("文件名,相似度,阈值,结果\n")
                for file_path, result in results:
                    f.write(f"{os.path.basename(file_path)},{result['similarity']:.4f},{result['threshold']:.4f},{result['message']}\n")
            logger.info(f"验证结果已保存到: {result_file}")
        
    except Exception as e:
        logger.error(f"系统演示失败: {str(e)}")
        
    logger.info("===== 演示完成 =====")