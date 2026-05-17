import re
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
log_file = "training_log.txt"  # 请修改为实际路径
epochs = []
train_loss = []
val_loss = []
val_acc = []

with open(log_file, "r") as f:
    for line in f:
        # 匹配格式: Epoch   1 | Train Loss: 1.6272 | Val Loss: 1.2315 | Val Acc: 0.6478
        match = re.search(r"Epoch\s+(\d+).*Train Loss:\s+([\d\.]+).*Val Loss:\s+([\d\.]+).*Val Acc:\s+([\d\.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            val_acc.append(float(match.group(4)))

# 找到最佳验证准确率对应的轮次
best_idx = np.argmax(val_acc)
best_epoch = epochs[best_idx]
best_acc = val_acc[best_idx]
best_val_loss = val_loss[best_idx]

print(f"最佳模型保存于 Epoch {best_epoch}, Val Acc = {best_acc:.4f}, Val Loss = {best_val_loss:.4f}")

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：损失曲线
ax1.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=1.5)
ax1.plot(epochs, val_loss, label='Val Loss', color='orange', linewidth=1.5)
ax1.scatter(best_epoch, best_val_loss, color='red', s=50, zorder=5, label=f'Best Val Loss (epoch {best_epoch})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# 右图：准确率曲线
ax2.plot(epochs, val_acc, label='Val Accuracy', color='green', linewidth=1.5)
ax2.scatter(best_epoch, best_acc, color='red', s=50, zorder=5, label=f'Best Val Acc = {best_acc:.4f} (epoch {best_epoch})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()