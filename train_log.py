# This file generate chart for latest training log --Maintainer: Jiming Yang

import re
import matplotlib.pyplot as plt
import numpy as np

# read log data
log_file = "training_log.txt" 
epochs = []
train_loss = []
val_loss = []
val_acc = []

with open(log_file, "r") as f:
    for line in f:
        
        match = re.search(r"Epoch\s+(\d+).*Train Loss:\s+([\d\.]+).*Val Loss:\s+([\d\.]+).*Val Acc:\s+([\d\.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            val_acc.append(float(match.group(4)))


best_idx = np.argmax(val_acc)
best_epoch = epochs[best_idx]
best_acc = val_acc[best_idx]
best_val_loss = val_loss[best_idx]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


ax1.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=1.5)
ax1.plot(epochs, val_loss, label='Val Loss', color='orange', linewidth=1.5)
ax1.scatter(best_epoch, best_val_loss, color='red', s=50, zorder=5, label=f'Best Val Loss (epoch {best_epoch})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)


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

# This file generate chart for latest training log --Maintainer: Jiming Yang

