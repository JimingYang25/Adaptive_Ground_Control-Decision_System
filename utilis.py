import numpy as np

def extract_features(buffer):
    """
    buffer: list of 10-element lists, 长度 = K (30)
    返回: 40维特征向量 (numpy array)
    """
    buf = np.array(buffer)   # (K, 10)
    feat = []
    # 当前帧（最新）
    feat.extend(buf[-1])
    # 均值
    feat.extend(np.mean(buf, axis=0))
    # 标准差
    feat.extend(np.std(buf, axis=0))
    # 差分（当前帧 - 前一帧）
    diff = buf[-1] - buf[-2] if len(buf) >= 2 else np.zeros(10)
    feat.extend(diff)
    return np.array(feat, dtype=np.float32)