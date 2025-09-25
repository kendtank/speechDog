# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/22 下午3:23
@Author  : Kend
@FileName: train_triplet.py
@Software: PyCharm
@modifier:
"""


# -*- coding: utf-8 -*-
"""
Triplet Loss Training for Dog Bark Voiceprint Verification
"""

import os
import tensorflow as tf
from triplet_dataset import TripletDataset

# -------------------------
# 模型定义
# -------------------------
def build_embedding(input_shape=(40, 40, 1), embedding_dim=64):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    emb = tf.keras.layers.Dense(embedding_dim)(x)
    emb = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(emb)  # L2 normalize
    return tf.keras.Model(inputs, emb, name="EmbeddingNet")

# -------------------------
# Triplet Loss
# -------------------------
def triplet_loss(a, p, n, margin=0.3):
    """Triplet Loss: max(d(a,p) - d(a,n) + margin, 0)"""
    d_ap = tf.reduce_sum(tf.square(a - p), axis=1)
    d_an = tf.reduce_sum(tf.square(a - n), axis=1)
    loss = tf.maximum(d_ap - d_an + margin, 0.0)
    return tf.reduce_mean(loss)

# -------------------------
# 训练循环
# -------------------------
def train(root_dir="logmel", epochs=10, batch_size=32, embedding_dim=64, export_path="dog_triplet.tflite"):
    dataset = TripletDataset(root_dir, batch_size=batch_size)
    net = build_embedding(embedding_dim=embedding_dim)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    # 日志
    summary_writer = tf.summary.create_file_writer("logs/triplet")

    step = 0
    for epoch in range(epochs):
        for (x, _) in dataset:
            anchors, positives, negatives = x
            with tf.GradientTape() as tape:
                a_emb = net(anchors, training=True)
                p_emb = net(positives, training=True)
                n_emb = net(negatives, training=True)
                loss = triplet_loss(a_emb, p_emb, n_emb)

            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

            # 写日志
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=step)
            step += 1

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.numpy():.4f}")

    # 保存 tf-lite
    converter = tf.lite.TFLiteConverter.from_keras_model(net)
    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ 模型已导出 {export_path}")

if __name__ == "__main__":
    train(root_dir="logmel", epochs=10, batch_size=32)


"""
# 用 logmel 特征训练
python train_triplet.py --root_dir logmel --epochs 30 --batch_size 32

# 用 mfcc 特征训练
python train_triplet.py --root_dir mfcc --epochs 30 --batch_size 32


TripletDataset 保证每个 batch 里一定有 A/P/N 组合；
Loss 是手写的 triplet loss，margin 可调（0.2–0.5 常用）；
导出 tf-lite 可直接端侧测试；
日志写到 logs/triplet，用 tensorboard --logdir logs/triplet 可看训练曲线。
"""