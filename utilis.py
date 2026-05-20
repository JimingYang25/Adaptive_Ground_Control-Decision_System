# This file provide input extraction --Maintainer: Jiming Yang

import numpy as np

def extract_features(buffer):
    """
    buffer: list of 10-element lists, length = K (default = 30)
    return: 40-dim features_array (numpy array)
    """
    buf = np.array(buffer)   # (K, 10)
    feat = []
    # current frame
    feat.extend(buf[-1])
    # mean value
    feat.extend(np.mean(buf, axis=0))
    # standard subtraction
    feat.extend(np.std(buf, axis=0))
    # differentiation (current_frame - last_frame)
    diff = buf[-1] - buf[-2] if len(buf) >= 2 else np.zeros(10)
    feat.extend(diff)
    return np.array(feat, dtype=np.float32)

# This file provide input extraction --Maintainer: Jiming Yang