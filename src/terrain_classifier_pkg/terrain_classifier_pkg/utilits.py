import numpy as np


def extract_features(buffer):
    """从滑动窗口缓冲区提取40维特征"""
    buf = np.array(buffer, dtype=np.float32)  # (K, 10)
    feat = []
    feat.extend(buf[-1])                     # 当前帧
    feat.extend(np.mean(buf, axis=0))        # 均值
    feat.extend(np.std(buf, axis=0))         # 标准差
    # 差分（确保 buffer 至少有两帧）
    if len(buf) >= 2:
        diff = buf[-1] - buf[-2]
    else:
        diff = np.zeros(10)
    feat.extend(diff)
    return np.array(feat, dtype=np.float32)