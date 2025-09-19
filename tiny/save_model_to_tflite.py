# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/19 下午5:20
@Author  : Kend
@FileName: save_model_to_tflite.py
@Software: PyCharm
@modifier:
"""


import tensorflow as tf

# ================== 配置 ==================
SAVED_MODEL_DIR = "tiny_cnn_bark_tf_savedmodel"  # 你的 SavedModel 路径
TFLITE_MODEL_PATH = "tiny_cnn_bark.tflite"       # 转换后保存路径

# ================== 创建转换器 ==================
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

# 可选优化（权重量化，不改输入输出类型）
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# ================== 转换 ==================
tflite_model = converter.convert()

# ================== 保存 TFLite ==================
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ 模型已成功转换为 TFLite: {TFLITE_MODEL_PATH}")
