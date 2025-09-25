# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/20 下午6:17
@Author  : Kend
@FileName: model_embedding.py
@Software: PyCharm
@modifier:
"""


"""
tinyML方向-判断是不是某只狗的声音（类似人声的说话人验证）。
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_tiny_cnn(input_shape=(64, 64, 1), emb_dim=64, num_classes=5):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    emb = layers.Dense(emb_dim, activation=None, name="embedding")(x)
    norm_emb = tf.nn.l2_normalize(emb, axis=1)   # 归一化 embedding (声纹验证用)

    outputs = layers.Dense(num_classes, activation="softmax")(norm_emb)

    model = models.Model(inputs, [outputs, norm_emb])
    return model
