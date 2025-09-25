# -*- coding: utf-8 -*-
"""
@Time    : 2025/9/25 下午2:08
@Author  : Kend
@FileName: train.py
@Software: PyCharm
@modifier:
"""

# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from model import TinyDogEmbeddingNet
from dataset import create_train_val_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EMBEDDING_DIM = 16


def mine_hard_triplets(embeddings, labels, margin=0.2):
    """在线 hard triplet mining"""
    n = len(embeddings)
    triplets = []
    for i in range(n):
        pos_dists = []
        neg_dists = []
        for j in range(n):
            if i == j: continue
            dist = 1 - np.dot(embeddings[i], embeddings[j])  # 1 - cosine
            if labels[i] == labels[j]:
                pos_dists.append((j, dist))
            else:
                neg_dists.append((j, dist))

        if not pos_dists or not neg_dists: continue

        # 最难正样本（距离最大）
        pos_idx = max(pos_dists, key=lambda x: x[1])[0]
        # 最难负样本（距离最小）
        neg_idx = min(neg_dists, key=lambda x: x[1])[0]

        # 满足 triplet 条件才加入
        if 1 - np.dot(embeddings[i], embeddings[pos_idx]) + margin > 1 - np.dot(embeddings[i], embeddings[neg_idx]):
            triplets.append((i, pos_idx, neg_idx))
    return triplets


def train():
    # 加载数据
    train_dataset, val_dataset, train_dogs, val_dogs = create_train_val_datasets(
        r"D:\kend\myPython\speechDog-master\datasets\dog_tiny_verification",
        val_ratio=0.3
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型 & 优化器
    model = TinyDogEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_eer = 1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_mel, batch_labels in train_loader:
            batch_mel, batch_labels = batch_mel.to(DEVICE), batch_labels.to(DEVICE)

            embeddings = model(batch_mel)

            # 在线 mining（简化：只在 batch 内找）
            emb_np = embeddings.detach().cpu().numpy()
            labels_np = batch_labels.cpu().numpy()
            triplets = mine_hard_triplets(emb_np, labels_np)

            if len(triplets) == 0:
                loss = 0.0 * embeddings.sum()  # 关键修复！
            else:
                anchor_idx, pos_idx, neg_idx = zip(*triplets)
                anchor = embeddings[torch.tensor(anchor_idx, device=DEVICE)]
                positive = embeddings[torch.tensor(pos_idx, device=DEVICE)]
                negative = embeddings[torch.tensor(neg_idx, device=DEVICE)]

                pos_dist = 1 - torch.sum(anchor * positive, dim=1)
                neg_dist = 1 - torch.sum(anchor * negative, dim=1)
                loss = torch.mean(torch.clamp(pos_dist - neg_dist + 0.2, min=0.0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        # 验证（计算 EER）
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            all_embs, all_labels = [], []
            with torch.no_grad():
                for mel, labels in val_loader:
                    embs = model(mel.to(DEVICE))
                    all_embs.append(embs.cpu().numpy())
                    all_labels.append(labels.numpy())
            all_embs = np.vstack(all_embs)
            all_labels = np.hstack(all_labels)

            # 计算所有对的相似度
            sim_matrix = cosine_similarity(all_embs)
            scores, labels = [], []
            for i in range(len(all_embs)):
                for j in range(i + 1, len(all_embs)):
                    scores.append(sim_matrix[i, j])
                    labels.append(1 if all_labels[i] == all_labels[j] else 0)

            # 简易 EER 估计
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            fnr = 1 - tpr
            eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
            eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

            print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}, Val EER: {eer:.4f}")

            if eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), "best_dog_embedding.pth")
                print(f"✅ New best model saved! EER: {eer:.4f}")

    # 导出为 TFLite（需先转 ONNX 或使用 TensorFlow）
    print("Training done. Now export to TFLite...")


if __name__ == "__main__":
    train()