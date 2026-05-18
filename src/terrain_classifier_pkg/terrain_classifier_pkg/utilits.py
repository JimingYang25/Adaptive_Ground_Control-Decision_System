#Provide imu_data pre-procession function -- Maintainer: Jiming Yang

import numpy as np

def extract_features(buffer):
    """Extract 40-dim features from sliding window buffer"""
    buf = np.array(buffer, dtype=np.float32)  # (K, 10)
    feat = []
    feat.extend(buf[-1])                     # current frame
    feat.extend(np.mean(buf, axis=0))        # mean value of K frames
    feat.extend(np.std(buf, axis=0))         # standard substract of K frames
    # differentiation
    if len(buf) >= 2:
        diff = buf[-1] - buf[-2]
    else:
        diff = np.zeros(10)
    feat.extend(diff)
    return np.array(feat, dtype=np.float32)

#Provide imu_data pre-procession function -- Maintainer: Jiming Yang
