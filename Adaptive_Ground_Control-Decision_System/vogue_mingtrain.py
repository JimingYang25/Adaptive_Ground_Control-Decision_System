import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ------------------------------
# Model（Vogue_Ming）
# ------------------------------
class Vogue_Ming(nn.Module):
    """Multitask MLP:Shared labelencoder + Classification header + Regression header(slope,roughness,height variation)"""
    def __init__(self, input_dim=40, hidden_dims=[480,240,120,40], num_classes=9, dropout=0.4):
        super().__init__()
        self.frame_branch=nn.Sequential(nn.Linear(10,60),nn.BatchNorm1d(60),nn.ReLU(),nn.Dropout(dropout),)
        self.window_branch=nn.Sequential(nn.Linear(30,180),nn.BatchNorm1d(180),nn.ReLU(),nn.Dropout(dropout))
        # shared layers
        layers = []
        prev_dim = 240
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.shared = nn.Sequential(*layers)
        
        # classification header
        self.class_head = nn.Linear(prev_dim, num_classes)
        # 回归头
        self.slope_head = nn.Linear(prev_dim, 1)
        self.rough_head = nn.Linear(prev_dim, 1)
        self.elev_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        extract=torch.cat((self.frame_branch(x[:,:10]),self.window_branch(x[:,10:])),dim=1)
        feat = self.shared(extract)
        cls_logits = self.class_head(feat)
        slope = self.slope_head(feat)
        rough = self.rough_head(feat)
        elev = self.elev_head(feat)
        return cls_logits, slope, rough, elev

# ------------------------------
# 1. 加载数据
# ------------------------------
df = pd.read_csv('./Modified_Datasets/frame_multi_target.csv')

# 特征列（以 feat_ 开头）
feature_cols = [c for c in df.columns if c.startswith('feat_')]
X = df[feature_cols].values.astype(np.float32)

# 目标列
y_cls = df['label_class'].values.astype(np.int64)
y_slope = df['slope'].values.astype(np.float32)
y_rough = df['roughness'].values.astype(np.float32)
y_elev = df['elevation_change'].values.astype(np.float32)
groups = df['series_id'].values

# 类别数
num_classes = len(np.unique(y_cls))
print(f"类别数: {num_classes}")

# 标签编码器（用于推理时输出字符串，这里仅保存）
le = LabelEncoder()
le.fit(y_cls)   # 用整数标签拟合（实际逆变换只能得到数字，但格式统一）

# ------------------------------
# 2. 划分训练/验证集（分组）
# ------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y_cls, groups=groups))

X_train_raw, X_val_raw = X[train_idx], X[val_idx]
y_cls_train, y_cls_val = y_cls[train_idx], y_cls[val_idx]
y_slope_train, y_slope_val = y_slope[train_idx], y_slope[val_idx]
y_rough_train, y_rough_val = y_rough[train_idx], y_rough[val_idx]
y_elev_train, y_elev_val = y_elev[train_idx], y_elev[val_idx]

# ------------------------------
# 3. 标准化（基于训练集）
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

# ------------------------------
# 4. 转换为 PyTorch 张量
# ------------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)

y_cls_train_t = torch.tensor(y_cls_train, dtype=torch.long)
y_cls_val_t = torch.tensor(y_cls_val, dtype=torch.long)

y_slope_train_t = torch.tensor(y_slope_train, dtype=torch.float32).view(-1,1)
y_slope_val_t = torch.tensor(y_slope_val, dtype=torch.float32).view(-1,1)
y_rough_train_t = torch.tensor(y_rough_train, dtype=torch.float32).view(-1,1)
y_rough_val_t = torch.tensor(y_rough_val, dtype=torch.float32).view(-1,1)
y_elev_train_t = torch.tensor(y_elev_train, dtype=torch.float32).view(-1,1)
y_elev_val_t = torch.tensor(y_elev_val, dtype=torch.float32).view(-1,1)

batch_size = 256
train_dataset = TensorDataset(X_train_t, y_cls_train_t, y_slope_train_t, y_rough_train_t, y_elev_train_t)
val_dataset = TensorDataset(X_val_t, y_cls_val_t, y_slope_val_t, y_rough_val_t, y_elev_val_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# 5. 初始化模型、损失函数、优化器
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Vogue_Ming().to(device)

cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
reg_criterion = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ------------------------------
# 6. 训练循环
# ------------------------------
epochs = 160
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for Xb, cls_b, slope_b, rough_b, elev_b in train_loader:
        Xb = Xb.to(device)
        cls_b = cls_b.to(device)
        slope_b = slope_b.to(device)
        rough_b = rough_b.to(device)
        elev_b = elev_b.to(device)

        optimizer.zero_grad()
        cls_logits, pred_slope, pred_rough, pred_elev = model(Xb)

        loss_cls = cls_criterion(cls_logits, cls_b)
        loss_slope = reg_criterion(pred_slope, slope_b)
        loss_rough = reg_criterion(pred_rough, rough_b)
        loss_elev = reg_criterion(pred_elev, elev_b)

        total_batch = loss_cls + 0.5 * (loss_slope + loss_rough + loss_elev)
        total_batch.backward()
        optimizer.step()
        total_loss += total_batch.item()

    # 验证
    model.eval()
    val_loss = 0.0
    val_cls_correct = 0
    val_total = 0
    with torch.no_grad():
        for Xb, cls_b, slope_b, rough_b, elev_b in val_loader:
            Xb = Xb.to(device)
            cls_b = cls_b.to(device)
            slope_b = slope_b.to(device)
            rough_b = rough_b.to(device)
            elev_b = elev_b.to(device)

            cls_logits, pred_slope, pred_rough, pred_elev = model(Xb)
            loss_cls = cls_criterion(cls_logits, cls_b)
            loss_slope = reg_criterion(pred_slope, slope_b)
            loss_rough = reg_criterion(pred_rough, rough_b)
            loss_elev = reg_criterion(pred_elev, elev_b)
            val_loss += (loss_cls + 0.5*(loss_slope+loss_rough+loss_elev)).item()

            _, pred_cls = torch.max(cls_logits, 1)
            val_total += cls_b.size(0)
            val_cls_correct += (pred_cls == cls_b).sum().item()

    val_loss_avg = val_loss / len(val_loader)
    val_acc = val_cls_correct / val_total
    print(f"Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step(val_loss_avg)

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), './Models/multi_task_best.pth')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("早停触发")
            break

print("训练完成，最佳模型已保存")

# ------------------------------
# 7. 保存标准化器和标签编码器
# ------------------------------
import os
os.makedirs('./Models', exist_ok=True)
joblib.dump(scaler, './Models/scaler.pkl')
joblib.dump(le, './Models/label_encoder.pkl')
print("标准化器和标签编码器已保存")