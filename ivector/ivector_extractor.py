import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import logging
import os
import pickle

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IVectorExtractor')

class IVectorExtractor:
    def __init__(self, n_components=8, tv_dim=100, reg_covar=1e-4, max_iter=50, random_state=42):
        """
        i-vector提取器初始化
        
        参数:
        ------
        n_components: int, 高斯混合模型的分量数
        tv_dim: int, i-vector的维度
        reg_covar: float, 协方差矩阵正则化参数，增加可提高稳定性
        max_iter: int, GMM训练的最大迭代次数
        random_state: int, 随机种子，保证结果可重现
        """
        # 参数验证
        if n_components <= 0:
            raise ValueError("高斯分量数必须为正整数")
        if tv_dim <= 0:
            raise ValueError("i-vector维度必须为正整数")
        if reg_covar <= 0:
            raise ValueError("正则化参数必须为正数")
            
        self.n_components = n_components
        self.tv_dim = tv_dim
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        
        # 模型组件
        self.ubm = None            # 通用背景模型
        self.pca = None            # 用于近似total variability矩阵的PCA模型
        self.feat_dim = None       # 特征维度
        self.is_trained = False    # 训练状态标志

    def fit(self, X_list):
        """
        训练i-vector提取器
        
        参数:
        ------
        X_list: list, 特征序列列表，每个元素为[frames, n_feats]形状的MFCC特征
        
        返回值:
        -------
        self: IVectorExtractor实例
        """
        # 输入验证
        if not X_list:
            raise ValueError("训练数据不能为空")
        
        if not all(isinstance(X, np.ndarray) for X in X_list):
            raise TypeError("所有输入必须是numpy数组")
        
        # 获取特征维度
        self.feat_dim = X_list[0].shape[1]
        
        # 记录训练数据信息
        logger.info(f"开始训练i-vector提取器")
        logger.info(f"训练数据: {len(X_list)}个语音段")
        logger.info(f"特征维度: {self.feat_dim}")
        logger.info(f"配置参数: 高斯分量数={self.n_components}, i-vector维度={self.tv_dim}, 正则化参数={self.reg_covar}")
        
        # 第一步：训练UBM GMM
        logger.info("训练通用背景模型(UBM GMM)...")
        
        # 合并所有特征用于GMM训练
        try:
            feats = np.vstack(X_list)
            logger.debug(f"合并后的训练数据形状: {feats.shape}")
            
            # 确保样本量足够训练GMM
            min_samples_needed = self.n_components * 10
            if feats.shape[0] < min_samples_needed:
                logger.warning(f"警告: 训练样本数({feats.shape[0]})少于建议值({min_samples_needed})，可能导致过拟合")
            
            # 训练GMM
            self.ubm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                max_iter=self.max_iter,
                reg_covar=self.reg_covar,
                tol=1e-3,
                random_state=self.random_state,
                verbose=0
            )
            
            self.ubm.fit(feats)
            logger.info("UBM GMM训练完成")
        except Exception as e:
            logger.error(f"UBM GMM训练失败: {str(e)}")
            raise
        
        # 第二步：计算统计量并训练PCA以近似total variability矩阵
        logger.info("计算充分统计量并估计total variability矩阵...")
        
        try:
            # 计算每个语音段的超向量
            supervectors = []
            for i, X in enumerate(X_list):
                # 定期记录进度
                if (i + 1) % 100 == 0 or i == len(X_list) - 1:
                    logger.debug(f"处理语音段 {i + 1}/{len(X_list)}")
                
                # 计算后验概率
                post = self.ubm.predict_proba(X)
                
                # 计算零阶和一阶统计量
                N = post.sum(axis=0)
                F = np.dot(post.T, X)
                
                # 计算超向量并归一化
                supervector = F.flatten() / (N.sum() + 1e-8)
                supervectors.append(supervector)
            
            supervectors = np.array(supervectors)
            logger.debug(f"计算得到的超向量形状: {supervectors.shape}")
            
            # 确保tv_dim不超过超向量维度
            max_pca_dim = min(self.tv_dim, supervectors.shape[1])
            if max_pca_dim < self.tv_dim:
                logger.warning(f"调整i-vector维度: {self.tv_dim} → {max_pca_dim} (受限于超向量维度)")
                self.tv_dim = max_pca_dim
            
            # 训练PCA模型
            self.pca = PCA(n_components=self.tv_dim, random_state=self.random_state)
            self.pca.fit(supervectors)
            
            # 计算PCA解释方差比例
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA训练完成，解释方差比例: {explained_variance:.4f}")
            
            # 标记为已训练
            self.is_trained = True
            logger.info("i-vector提取器训练成功完成")
            
        except Exception as e:
            logger.error(f"Total variability矩阵估计失败: {str(e)}")
            raise
        
        return self

    def transform(self, X):
        """
        从单个语音段提取i-vector特征
        
        参数:
        ------
        X: np.ndarray, 形状为[frames, n_feats]的MFCC特征
        
        返回值:
        -------
        ivec: np.ndarray, 形状为[tv_dim]的i-vector特征
        """
        # 检查是否已训练
        if not self.is_trained:
            raise RuntimeError("必须先调用fit()方法训练提取器")
        
        # 输入验证
        if not isinstance(X, np.ndarray):
            raise TypeError("输入必须是numpy数组")
        
        if X.ndim != 2:
            raise ValueError("输入特征必须是二维数组: [frames, n_feats]")
        
        if X.shape[1] != self.feat_dim:
            raise ValueError(f"特征维度不匹配: 期望{self.feat_dim}，得到{X.shape[1]}")
        
        # 确保有足够的帧
        min_frames = 5
        if X.shape[0] < min_frames:
            logger.warning(f"警告: 输入帧数({X.shape[0]})少于建议值({min_frames})，可能影响提取结果")
        
        try:
            # 计算后验概率
            post = self.ubm.predict_proba(X)  # [frames, n_components]
            
            # 计算统计量
            N = post.sum(axis=0)               # 零阶统计量
            F = np.dot(post.T, X)              # 一阶统计量
            
            # 计算超向量
            supervector = F.flatten() / (N.sum() + 1e-8)
            
            # 使用PCA将超向量压缩为i-vector
            ivec = self.pca.transform(supervector.reshape(1, -1))[0]
            
            # L2归一化i-vector
            ivec_norm = np.linalg.norm(ivec)
            if ivec_norm > 1e-10:  # 避免除零
                ivec = ivec / ivec_norm
            
            return ivec
            
        except Exception as e:
            logger.error(f"i-vector提取失败: {str(e)}")
            raise
    
    def batch_transform(self, X_list):
        """
        批量提取多个语音段的i-vector特征
        
        参数:
        ------
        X_list: list, 特征序列列表，每个元素为[frames, n_feats]形状的MFCC特征
        
        返回值:
        -------
        ivectors: np.ndarray, 形状为[len(X_list), tv_dim]的i-vector特征矩阵
        """
        logger.info(f"批量提取 {len(X_list)} 个语音段的i-vector...")
        
        # 验证输入
        if not X_list:
            return np.array([])
            
        # 提取所有i-vectors
        ivectors = []
        for i, X in enumerate(X_list):
            # 定期记录进度
            if (i + 1) % 100 == 0 or i == len(X_list) - 1:
                logger.debug(f"处理语音段 {i + 1}/{len(X_list)}")
            
            # 提取单个i-vector
            ivec = self.transform(X)
            ivectors.append(ivec)
            
        return np.array(ivectors)
        
    def save(self, filepath):
        """
        保存模型到文件
        
        参数:
        ------
        filepath: str, 保存模型的文件路径
        """
        if not self.is_trained:
            logger.warning("警告: 尝试保存未训练的模型")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # 保存模型参数
            model_params = {
                'n_components': self.n_components,
                'tv_dim': self.tv_dim,
                'reg_covar': self.reg_covar,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
                'feat_dim': self.feat_dim,
                'ubm': self.ubm,
                'pca': self.pca,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_params, f)
                
            logger.info(f"模型已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            raise
            
    @classmethod
    def load(cls, filepath):
        """
        从文件加载模型
        
        参数:
        ------
        filepath: str, 模型文件路径
        
        返回值:
        -------
        extractor: IVectorExtractor实例
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"模型文件不存在: {filepath}")
                
            # 加载模型参数
            with open(filepath, 'rb') as f:
                model_params = pickle.load(f)
                
            # 创建实例
            extractor = cls(
                n_components=model_params['n_components'],
                tv_dim=model_params['tv_dim'],
                reg_covar=model_params['reg_covar'],
                max_iter=model_params['max_iter'],
                random_state=model_params['random_state']
            )
            
            # 恢复模型组件
            extractor.feat_dim = model_params['feat_dim']
            extractor.ubm = model_params['ubm']
            extractor.pca = model_params['pca']
            extractor.is_trained = model_params['is_trained']
            
            logger.info(f"模型已从 {filepath} 加载")
            return extractor
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
