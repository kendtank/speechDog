# -*- coding: utf-8 -*-
"""
@Time    : 2025/09/20
@Author  : Kend
@FileName: train_dog_embedding_tinyml.py
@Software: PyCharm
@Description: TinyML 狗吠声 embedding 训练脚本
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from datetime import datetime

# -------------------- 参数配置 --------------------
DATA_DIR = r"D:\kend\myPython\speechDog-master\datasets\compare_dog_mel_dataset"
BATCH_SIZE = 16
EPOCHS = 30
EMB_DIM = 64   # embedding 维度，可小于 64
RANDOM_SEED = 42

def load_mel_fixed(file, target_shape=(40,64)):
    mel = np.load(file)
    h, w = mel.shape
    th, tw = target_shape

    # pad 高度
    if h < th:
        pad_h = th - h
        mel = np.pad(mel, ((0,pad_h),(0,0)), mode="constant")
    else:
        mel = mel[:th,:]

    # pad 宽度
    if w < tw:
        pad_w = tw - w
        mel = np.pad(mel, ((0,0),(0,pad_w)), mode="constant")
    else:
        mel = mel[:,:tw]

    return mel




# -------------------- 数据准备 --------------------
X = []
y = []
classes = sorted(os.listdir(DATA_DIR))
class_to_idx = {c: i for i, c in enumerate(classes)}

for c in classes:
    folder = os.path.join(DATA_DIR, c)
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            mel = load_mel_fixed(os.path.join(folder, f))
            X.append(mel[..., np.newaxis])
            y.append(class_to_idx[c])


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 打乱并切分训练集/验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.33, random_state=RANDOM_SEED, stratify=y
)

print(f"数据集大小: {X.shape}, 标签数量: {len(classes)}")
print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

# -------------------- 模型定义 --------------------
def build_tiny_cnn(input_shape, num_classes, emb_dim=EMB_DIM):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, kernel_size=(3,3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(32, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    emb = layers.Dense(emb_dim, activation=None, name="norm_embedding")(x)
    norm_emb = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(emb)

    # 分类输出
    outputs = layers.Dense(num_classes, activation="softmax", name="dense")(norm_emb)

    model = models.Model(inputs, outputs)
    return model

model = build_tiny_cnn(X_train.shape[1:], num_classes=len(classes))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------- 训练 --------------------
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv"))

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[csv_logger]
)

# -------------------- 保存模型 --------------------
model_dir = os.path.join("saved_models", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(model_dir, exist_ok=True)
model.export(os.path.join(model_dir, "dog_embedding_model"))
# model_path = os.path.join(model_dir, "dog_embedding_model.h5")
# model.save(model_path)
print(f"模型训练完成并保存")

# -------------------- 训练完成 --------------------
