import os
import sys
import numpy as np
import logging
from one_class_dog_verification import OneClassDogIVectorSystem

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("test_one_class_system.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 创建示例音频数据（用于测试）
def create_demo_audio(output_dir):
    """
    创建用于测试的示例音频数据结构
    注意：这只是创建目录结构，实际音频文件需要用户提供
    """
    logger.info("创建测试目录结构...")
    
    # 创建目标狗注册目录
    target_dog_dir = os.path.join(output_dir, "enroll_target_dog")
    os.makedirs(target_dog_dir, exist_ok=True)
    
    # 创建目标狗测试目录
    target_test_dir = os.path.join(output_dir, "test_target_dog")
    os.makedirs(target_test_dir, exist_ok=True)
    
    # 创建非目标狗测试目录
    non_target_test_dir = os.path.join(output_dir, "test_non_target_dog")
    os.makedirs(non_target_test_dir, exist_ok=True)
    
    logger.info(f"测试目录已创建在: {output_dir}")
    logger.info("请将目标狗的注册音频放入: enroll_target_dog/")
    logger.info("请将目标狗的测试音频放入: test_target_dog/")
    logger.info("请将非目标狗的测试音频放入: test_non_target_dog/")
    
    # 返回创建的目录路径
    return target_dog_dir, target_test_dir, non_target_test_dir

# 检查目录中是否有音频文件
def check_audio_files(directory):
    """
    检查目录中是否包含.wav文件
    """
    if not os.path.exists(directory):
        return False, []
    
    wav_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if f.lower().endswith('.wav')]
    
    return len(wav_files) > 0, wav_files

# 运行系统测试
def run_system_test():
    """
    运行单类狗声纹验证系统的完整测试
    """
    global logger
    logger = setup_logging()
    logger.info("===== 单类狗声纹验证系统测试开始 =====")
    
    # 使用用户已有的数据目录（根据one_class_dog_verification.py中的配置）
    target_enroll_dir = "./youtube_wav/brakng_dog_datasets/dog1"
    target_test_dir = "./youtube_wav/test"
    
    # 如果测试目录不存在，尝试使用默认的测试目录结构
    if not os.path.exists(target_enroll_dir) or not os.path.exists(target_test_dir):
        logger.warning("未找到用户配置的数据目录，尝试创建并使用默认测试目录结构")
        test_data_dir = "test_data"
        target_enroll_dir, target_test_dir, non_target_test_dir = create_demo_audio(test_data_dir)
        
        # 检查注册目录
        has_enroll_files, enroll_files = check_audio_files(target_enroll_dir)
        if not has_enroll_files:
            logger.error(f"错误: 在{target_enroll_dir}中未找到任何.wav文件，请先添加目标狗的注册音频")
            logger.info("===== 测试失败，请提供音频文件后重新运行 ======")
            return False
    else:
        logger.info(f"使用用户配置的数据目录:")
        logger.info(f"  注册目录: {target_enroll_dir}")
        logger.info(f"  测试目录: {target_test_dir}")
        
        # 创建一个模拟的非目标测试目录（在实际应用中应该有单独的非目标狗音频）
        non_target_test_dir = target_test_dir
    
    # 检查注册目录
    has_enroll_files, enroll_files = check_audio_files(target_enroll_dir)
    if not has_enroll_files:
        logger.error(f"错误: 在{target_enroll_dir}中未找到任何.wav文件，请先添加目标狗的注册音频")
        logger.info("===== 测试失败，请提供音频文件后重新运行 ======")
        return False
    
    # 检查测试目录
    has_target_test_files, target_test_files = check_audio_files(target_test_dir)
    has_non_target_test_files, non_target_test_files = check_audio_files(non_target_test_dir)
    
    # 初始化系统
    logger.info("初始化单类狗声纹验证系统...")
    system = OneClassDogIVectorSystem(
        dog_name="target_dog",
        n_components=2,  # 小分量数适合测试
        tv_dim=8,        # 低维度适合测试
        threshold=0.4    # 初始阈值
    )
    
    try:
        # 训练系统
        logger.info(f"开始训练系统，使用{len(enroll_files)}个注册音频...")
        system.train(enroll_files)
        logger.info("系统训练完成")
        
        # 保存模型
        model_path = os.path.join("models", "test_one_class_model.pkl")
        logger.info(f"保存模型到: {model_path}")
        success = system.save_model(model_path)
        if not success:
            logger.warning("模型保存失败")
        
        # 加载模型（测试模型保存/加载功能）
        logger.info("加载已保存的模型...")
        loaded_system = OneClassDogIVectorSystem.load_model(model_path)
        logger.info("模型加载成功")
        
        # 验证目标狗的音频
        if has_target_test_files:
            logger.info(f"\n验证目标狗的音频文件 ({len(target_test_files)}个)...")
            target_results = []
            
            for test_file in target_test_files:
                result = loaded_system.verify(test_file)
                target_results.append(result)
                logger.info(f"文件: {os.path.basename(test_file)}, 相似度: {result['similarity']:.4f}, 结果: {result['message']}")
            
            # 计算目标狗验证的统计信息
            target_success_rate = np.mean([r['is_target'] for r in target_results])
            avg_target_similarity = np.mean([r['similarity'] for r in target_results])
            logger.info(f"目标狗验证统计: 成功率={target_success_rate:.2%}, 平均相似度={avg_target_similarity:.4f}")
        
        # 验证非目标狗的音频
        if has_non_target_test_files:
            logger.info(f"\n验证非目标狗的音频文件 ({len(non_target_test_files)}个)...")
            non_target_results = []
            
            for test_file in non_target_test_files:
                result = loaded_system.verify(test_file)
                non_target_results.append(result)
                logger.info(f"文件: {os.path.basename(test_file)}, 相似度: {result['similarity']:.4f}, 结果: {result['message']}")
            
            # 计算非目标狗验证的统计信息
            non_target_reject_rate = np.mean([not r['is_target'] for r in non_target_results])
            avg_non_target_similarity = np.mean([r['similarity'] for r in non_target_results])
            logger.info(f"非目标狗验证统计: 拒绝率={non_target_reject_rate:.2%}, 平均相似度={avg_non_target_similarity:.4f}")
        
        # 由于可能没有单独的非目标狗测试音频，我们修改阈值调整逻辑
        if has_target_test_files:
            # 使用测试目录中的部分文件作为伪非目标样本
            # 在实际应用中应该有单独的非目标狗音频
            pseudo_non_target_files = target_test_files[:len(target_test_files)//2] if len(target_test_files) > 2 else target_test_files
            
            logger.info("\n调整验证阈值...")
            threshold_result = loaded_system.tune_threshold(
                target_test_files,
                pseudo_non_target_files,
                target_frr=0.1  # 目标误拒率10%
            )
            
            logger.info(f"阈值调整结果:")
            logger.info(f"  新阈值: {threshold_result['threshold']:.4f}")
            logger.info(f"  误拒率(FRR): {threshold_result['frr']:.4%}")
            logger.info(f"  误纳率(FAR): {threshold_result['far']:.4%}")
            logger.info("注意: 由于没有单独的非目标狗音频，这里使用了伪非目标样本进行阈值调整")
        
        logger.info("\n===== 单类狗声纹验证系统测试完成 =====")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        logger.info("===== 测试失败 =====")
        return False

# 主函数
if __name__ == "__main__":
    # 创建models目录（如果不存在）
    os.makedirs("models", exist_ok=True)
    
    # 运行测试
    success = run_system_test()
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1)