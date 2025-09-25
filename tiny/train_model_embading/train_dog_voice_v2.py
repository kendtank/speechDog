# -*- coding: utf-8 -*-
"""
@Time    : 2025/09/22
@Author  : Kend & GPT
@FileName: train_dog_voice_v2.py
@Software: PyCharm
@Description: 声纹验证训练脚本 (支持特征/模型/损失参数选择 + Triplet Loss 自采样)
"""

import os
import argparse
import random
import numpy as np
import tensorflow as tf


# =====================
# 参数
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--feature", type=str, default="mfcc", choices=["logmel", "mfcc", "spectral"],
                    help="输入特征类型")
parser.add_argument("--arch", type=str, default="tdnn", choices=["cnn", "resnet", "cnn1d", "tdnn"],
                    help="模型架构")
parser.add_argument("--loss", type=str, default="triplet", choices=["triplet", "contrastive", "softmax"],
                    help="损失函数")
parser.add_argument("--data_root", type=str,
                    default=r"D:\kend\myPython\speechDog-master\datasets\dog_embedding_features", help="特征数据根目录")
parser.add_argument("--epochs", type=int, default=100)  # 增加训练轮数
parser.add_argument("--batch_size", type=int, default=32)  # 减小批次大小以提高训练稳定性
parser.add_argument("--embed_dim", type=int, default=128, help="embedding 维度")  # 增加嵌入维度
parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
parser.add_argument("--specaug", action="store_true", help="是否开启 SpecAugment 数据增强")
parser.add_argument("--lr", type=float, default=0.001, help="学习率")

args = parser.parse_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================
# 数据加载
# =====================
def load_data(feature_dir):
    X, y = [], []
    dog_ids = sorted(os.listdir(feature_dir))
    id_map = {dog_id: idx for idx, dog_id in enumerate(dog_ids)}

    for dog_id in dog_ids:
        dog_path = os.path.join(feature_dir, dog_id)
        for fname in os.listdir(dog_path):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(dog_path, fname))
                X.append(arr)
                y.append(id_map[dog_id])

    return np.array(X), np.array(y), len(dog_ids)


def make_triplets(X, y, num_triplets=1024):
    """ 构造 (Anchor, Positive, Negative) """
    triplets = []
    n_samples = len(X)

    for _ in range(num_triplets):
        anchor_idx = random.randint(0, n_samples - 1)
        anchor_label = y[anchor_idx]

        pos_candidates = np.where(y == anchor_label)[0]
        neg_candidates = np.where(y != anchor_label)[0]

        # 确保存在正样本和负样本
        if len(pos_candidates) < 2 or len(neg_candidates) == 0:
            continue

        pos_idx = random.choice(pos_candidates[pos_candidates != anchor_idx])
        neg_idx = random.choice(neg_candidates)

        triplets.append((X[anchor_idx], X[pos_idx], X[neg_idx]))

    if len(triplets) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a, p, n = zip(*triplets)
    return np.array(a), np.array(p), np.array(n)


class SpecAugment(tf.keras.layers.Layer):
    """SpecAugment: 时间遮挡 + 频率遮挡"""
    def __init__(self, freq_mask=10, time_mask=20, freq_mask_rate=0.2, time_mask_rate=0.2, **kwargs):
        super(SpecAugment, self).__init__(**kwargs)
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.freq_mask_rate = freq_mask_rate  # 频率遮蔽比例上限
        self.time_mask_rate = time_mask_rate  # 时间遮蔽比例上限

    def call(self, x, training=None):
        if training:
            # x: (B, T, F) or (T, F)
            f = tf.shape(x)[-1]
            t = tf.shape(x)[-2]

            # 计算基于比例的最大遮蔽尺寸
            max_freq_mask = tf.cast(tf.floor(tf.cast(f, tf.float32) * self.freq_mask_rate), tf.int32)
            max_time_mask = tf.cast(tf.floor(tf.cast(t, tf.float32) * self.time_mask_rate), tf.int32)
            
            # 频率遮挡 - 不超过频率维度的20%
            f_mask = tf.minimum(self.freq_mask, max_freq_mask)
            f_mask = tf.minimum(f_mask, f)  # 确保不超过总频率数
            f0 = tf.random.uniform([], 0, tf.maximum(1, f - f_mask), dtype=tf.int32)
            freq_mask = tf.range(f0, tf.minimum(f0 + f_mask, f))
            freq_updates = tf.zeros((tf.shape(freq_mask)[0],), dtype=x.dtype)
            x = tf.tensor_scatter_nd_update(
                x,
                indices=tf.expand_dims(freq_mask, axis=1),
                updates=tf.broadcast_to(freq_updates, [tf.shape(freq_mask)[0]])
            )

            # 时间遮挡 - 不超过时间维度的20%
            t_mask = tf.minimum(self.time_mask, max_time_mask)
            t_mask = tf.minimum(t_mask, t)  # 确保不超过总时间帧数
            t0 = tf.random.uniform([], 0, tf.maximum(1, t - t_mask), dtype=tf.int32)
            time_mask = tf.range(t0, tf.minimum(t0 + t_mask, t))
            time_updates = tf.zeros((tf.shape(time_mask)[0],), dtype=x.dtype)
            x = tf.tensor_scatter_nd_update(
                x,
                indices=tf.expand_dims(time_mask, axis=1),
                updates=tf.broadcast_to(time_updates, [tf.shape(time_mask)[0]])
            )
        return x


# =====================
# 模型定义
# =====================

def se_block(inputs, ratio=8):
    """Squeeze-and-Excitation block (纯 Keras 写法)"""
    filters = inputs.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    se = tf.keras.layers.Dense(filters // ratio, activation="relu")(se)
    se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)
    se = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(se)
    x = tf.keras.layers.Multiply()([inputs, se])
    return x


def tdnn_block(x, out_dim, context_size=5, dilation=1, activation="relu"):
    """TDNN Block (1D Conv with dilation)"""
    x = tf.keras.layers.Conv1D(out_dim, context_size,
                               dilation_rate=dilation,
                               padding="same",
                               activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def build_model(input_shape, arch="cnn1d", embed_dim=64, num_classes=None, use_specaug=False):
    """
    input_shape: (time, freq, 1)
    arch: cnn1d / resnet / cnn / tdnn
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if use_specaug:
        # 如果是 2D 特征 (freq, time, 1) → reshape 成 (time, freq)
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(x)
        x = SpecAugment()(x)
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(x)

    if arch == "cnn1d":
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(inputs)  # (T, F)
        x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)  # 增加通道数
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)  # 增加通道数
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    elif arch == "cnn":
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)  # 增加通道数
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    elif arch == "resnet":
        # 为ResNet调整输入形状，确保是3通道
        if input_shape[-1] != 3:
            x = tf.keras.layers.Conv2D(3, (1, 1), padding='same')(inputs)  # 转换为3通道
        else:
            x = inputs
        base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=x)
        x = base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    elif arch == "tdnn":
        # Reshape to (time, freq)
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1]))(inputs)

        # TDNN Blocks with dilation
        x = tdnn_block(x, 128, context_size=5, dilation=1)  # 增加通道数
        x = tdnn_block(x, 256, context_size=3, dilation=2)  # 增加通道数
        x = tdnn_block(x, 256, context_size=3, dilation=3)

        # SE-Block
        x = se_block(x)

        # Pooling (统计池化)
        mean = tf.keras.layers.GlobalAveragePooling1D()(x)
        std = tf.keras.layers.Lambda(lambda t: tf.math.reduce_std(t, axis=1))(x)
        x = tf.keras.layers.Concatenate()([mean, std])

        x = tf.keras.layers.Dense(512, activation="relu")(x)  # 增加全连接层维度

    # embedding 层
    embeddings = tf.keras.layers.Dense(embed_dim, activation=None, name="embedding")(x)
    embeddings = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embeddings)

    if args.loss == "softmax" and num_classes is not None:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(embeddings)
        return tf.keras.Model(inputs, outputs, name=f"{arch}_softmax")
    else:
        return tf.keras.Model(inputs, embeddings, name=f"{arch}_embed")


# =====================
# 损失函数
# =====================
def triplet_loss_fn(anchor, positive, negative, margin=0.2):  # 减小margin以提高区分度
    d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(d_ap - d_an + margin, 0.0)
    return tf.reduce_mean(loss)


def contrastive_loss_fn(y_true, d, margin=1.0):
    return tf.reduce_mean(y_true * tf.square(d) + (1 - y_true) * tf.square(tf.maximum(margin - d, 0)))


# ---------------------
# 验证集准确率函数
# ---------------------
def compute_val_acc(model, X_val, y_val, num_samples=200):
    correct = 0
    total = 0
    n_val = len(X_val)
    
    # 确保有足够的正负样本对
    if n_val < 3:
        return 0.0
        
    for _ in range(num_samples):
        a_idx = np.random.randint(0, n_val)
        a_label = y_val[a_idx]
        pos_idxs = np.where(y_val == a_label)[0]
        if len(pos_idxs) < 2:
            continue
        p_idx = np.random.choice(pos_idxs[pos_idxs != a_idx])
        neg_idxs = np.where(y_val != a_label)[0]
        if len(neg_idxs) == 0:
            continue
        n_idx = np.random.choice(neg_idxs)

        emb_a = model(np.expand_dims(X_val[a_idx],0), training=False)
        emb_p = model(np.expand_dims(X_val[p_idx],0), training=False)
        emb_n = model(np.expand_dims(X_val[n_idx],0), training=False)

        d_pos = tf.reduce_sum(tf.square(emb_a - emb_p))
        d_neg = tf.reduce_sum(tf.square(emb_a - emb_n))
        if d_pos < d_neg:
            correct += 1
        total += 1
    if total == 0:
        return 0.0
    return correct / total


def compute_val_acc_batch(model, X_val, y_val, num_samples=200):
    n_val = len(X_val)
    correct = 0
    total = 0
    
    # 确保有足够的样本
    if n_val < 3:
        return 0.0

    # 随机采样三元组索引
    for _ in range(num_samples):
        a_idx = np.random.randint(0, n_val)
        a_label = y_val[a_idx]

        pos_idxs = np.where(y_val == a_label)[0]
        if len(pos_idxs) < 2:
            continue
        p_idx = np.random.choice(pos_idxs[pos_idxs != a_idx])

        neg_idxs = np.where(y_val != a_label)[0]
        if len(neg_idxs) == 0:
            continue
        n_idx = np.random.choice(neg_idxs)

        total += 1

        # 一次前向推理三个样本
        batch = np.stack([X_val[a_idx], X_val[p_idx], X_val[n_idx]], axis=0)
        emb = model(batch, training=False).numpy()

        d_pos = np.sum((emb[0]-emb[1])**2)
        d_neg = np.sum((emb[0]-emb[2])**2)
        if d_pos < d_neg:
            correct += 1

    return correct / total if total > 0 else 0.0




# =====================
# 主训练逻辑
# =====================
def main():
    # 数据路径
    feature_dir = os.path.join(args.data_root, args.feature)
    X, y, num_classes = load_data(feature_dir)

    # 检查数据是否加载成功
    if len(X) == 0:
        print("❌ 未加载到任何数据，请检查数据路径和格式")
        return

    # 统一输入 shape (freq, time, 1)
    X = np.expand_dims(X, -1)
    input_shape = X.shape[1:]

    # 构建模型
    model = build_model(input_shape, args.arch, args.embed_dim, num_classes, use_specaug=args.specaug)

    optimizer = tf.keras.optimizers.Adam(args.lr)  # 使用可配置的学习率

    # ---------------------
    # 训练循环修改
    # ---------------------
    # 划分训练/验证
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    if args.loss == "softmax":
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, y_val))
    else:
        best_val_acc = 0.0
        for epoch in range(args.epochs):
            a, p, n = make_triplets(X_train, y_train, num_triplets=max(1, len(X_train) // 2))
            
            # 检查是否生成了有效的三元组
            if len(a) == 0:
                print(f"[Epoch {epoch + 1}/{args.epochs}] 警告：未能生成有效三元组，跳过本轮训练")
                continue
                
            with tf.GradientTape() as tape:
                emb_a = model(a, training=True)
                emb_p = model(p, training=True)
                emb_n = model(n, training=True)

                if args.loss == "triplet":
                    loss = triplet_loss_fn(emb_a, emb_p, emb_n)
                elif args.loss == "contrastive":
                    d_pos = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
                    y_true_pos = tf.ones_like(d_pos)
                    loss_pos = contrastive_loss_fn(y_true_pos, d_pos)

                    d_neg = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
                    y_true_neg = tf.zeros_like(d_neg)
                    loss_neg = contrastive_loss_fn(y_true_neg, d_neg)

                    loss = (loss_pos + loss_neg) / 2

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 间隔五次验证准确率
            if (epoch + 1) % 5 == 0:
                val_acc = compute_val_acc_batch(model, X_val, y_val)
                print(f"[Epoch {epoch + 1}/{args.epochs}] Loss={loss.numpy():.4f} | Val_Acc={val_acc:.4f}")
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    saved_model_path = os.path.join(args.save_dir, f"{args.feature}_{args.arch}_{args.loss}_best_savedmodel")
                    model.export(saved_model_path)
                    print(f"✅ 保存最佳模型 Val_Acc={val_acc:.4f}")
            else:
                print(f"[Epoch {epoch + 1}/{args.epochs}] Loss={loss.numpy():.4f}")


    # ---------------------
    # 保存模型修改
    # ---------------------
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存为SavedModel格式，这是推荐的TensorFlow格式
    saved_model_path = os.path.join(args.save_dir, f"{args.feature}_{args.arch}_{args.loss}_savedmodel")
    model.export(saved_model_path)
    print(f"✅ SavedModel 已导出到 {saved_model_path}")

    # 导出 TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    tflite_path = saved_model_path + ".tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite 已保存到 {tflite_path}")


if __name__ == "__main__":
    main()


"""
# Triplet Loss + TDNN + SE Block + MFCC (推荐配置)
python train_dog_voice_v2.py --feature mfcc --arch tdnn --loss triplet --specaug

# 其他训练命令示例:
# Triplet Loss + CNN + logmel
python train_dog_voice_v2.py --feature mfcc --arch cnn1d --loss triplet 

python train_dog_voice_v2.py --feature spectral --arch cnn1d --loss triplet

# Contrastive Loss + ResNet + mfcc
python train_dog_voice_v2.py --feature mfcc --arch resnet --loss contrastive

# Softmax 分类 (多狗识别任务)
python train_dog_voice_v2.py --feature spectral --arch cnn --loss softmax

# Triplet Loss + TDNN + SE Block + MFCC
python train_dog_voice_v2.py --feature mfcc --arch tdnn --loss triplet 

python train_dog_voice_v2.py --feature mfcc --arch tdnn --loss triplet  --specaug

# Contrastive Loss + TDNN
python train_dog_voice_v2.py --feature logmel --arch tdnn --loss contrastive

这样做的好处：
时序建模能力接近 GRU/LSTM，但用的全是 Conv → 100% TFLite 兼容。
SE-Block 能自动抑制噪声频段，提升鲁棒性。
统计池化 (mean+std pooling) → 标准的 d-vector 风格。
"""



"""
| **维度**       | **CNN**   | **ResNet**         | **CNN+GRU**   |
| ------------ | --------- | ------------------ | ------------- |
| **特征提取能力**   | 基础局部特征    | 深层次抽象特征            | 时序+局部特征       |
| **计算开销**     | ⭐（轻量）     | ⭐⭐⭐⭐（大）            | ⭐⭐（中等）        |
| **对数据规模的需求** | 小数据也能训    | 需要较大数据，否则过拟合       | 中等（比纯 CNN 要多） |
| **泛化能力**     | 一般        | 强                  | 较强，特别适合声音任务   |
| **适合的场景**    | 端侧部署、快速实验 | 高精度 baseline、大规模训练 | 声纹/音频验证（推荐）   |


| **损失函数**             | **训练目标**   | **优点**                 | **缺点**            | **适用场景**                |
| -------------------- | ---------- | ---------------------- | ----------------- | ----------------------- |
| **Softmax**          | 分类（固定类别）   | 收敛快，精度高                | 泛化差，新个体无法识别       | 固定类目任务（狗 A / 狗 B / 狗 C） |
| **Triplet Loss**     | A 近 P，远离 N | 直接优化 embedding 空间，适合验证 | 采样难度大，需要平衡        | Open-set 验证（声纹/宠物重识别）   |
| **Contrastive Loss** | 正样本近，负样本远  | pairwise 训练更稳定，小数据友好   | 空间结构不如 Triplet 精细 | 数据量较少时的验证场景             |

首选：arch=tdnn + loss=triplet → 最符合声纹验证逻辑。

备用：arch=cnn + loss=contrastive → 数据不够时更稳。
"""