#This file for datasets procession --Maintainer: Jiming Yang

import pandas as pd
import numpy as np
from scipy import signal

# ------------------------------
# 1. Read orin_dataset
# ------------------------------
X = pd.read_csv('./Dataset/X_train.csv')
y = pd.read_csv('./Dataset/y_train.csv')
X = X.sort_values(['series_id', 'measurement_number'])

feature_cols = [c for c in X.columns if c not in ['row_id', 'series_id', 'measurement_number']]

# ------------------------------
# 2. Build protocol data (samples, 128, 10)
# ------------------------------
X_seq = []
series_ids = []
for sid, group in X.groupby('series_id'):
    mat = group[feature_cols].values
    if mat.shape[0] == 128:
        X_seq.append(mat)
        series_ids.append(sid)
X_seq = np.array(X_seq, dtype=np.float32)  # (n_segments, 128, 10)

# label mapping
y_filtered = y[y['series_id'].isin(series_ids)].drop_duplicates('series_id')
y_aligned = y_filtered.set_index('series_id').loc[series_ids]
surface_names = y_aligned['surface'].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(surface_names)
num_classes = len(le.classes_)

# Define the mapping from categories to reference values (empirical values)
# slope(degree), Roughness(0~1), Elevation change(m/s)
surface_base = {
    'carpet': (0.0, 0.3, 0.0),
    'concrete': (0.0, 0.1, 0.0),
    'fine_concrete': (0.0, 0.05, 0.0),
    'soft_tiles': (0.0, 0.35, 0.0),
    'tiled': (0.0, 0.25, 0.0),
    'soft_pvc': (0.0, 0.2, 0.0),
    'hard_tiles_large_space': (0.0, 0.15, 0.0),
    'grass': (1.5, 0.5, 0.02),      
    'gravel': (2.0, 0.8, 0.05),
    # others 
}

# Auxiliary Function: Quaternion to Euler Angle (Pitch Angle)
def quat_to_pitch(qw, qx, qy, qz):
    
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    return np.arcsin(sinp)

# ------------------------------
# 3. Generate frame-level samples (including continuous labels)
# ------------------------------
K = 30   # Window_length
window_stride = 1  

X_feat = []   # 40-dim features
y_class = []  # classes labels
y_slope = []  # slope
y_rough = []  # roughness
y_elev = []   # elevation change
group_feat = []

for idx, (sid, sample) in enumerate(zip(series_ids, X_seq)):
    surface = surface_names[idx]
    base_slope, base_rough, base_elev = surface_base.get(surface, (0.0, 0.2, 0.0))
    
    # Precompute some sequence features for dynamic adjustment
    # Extract the linear acceleration along the z-axis (9th column, index 8)
    acc_z = sample[:, 8]   # (128,)

    ang_vel = sample[:, 4:7]  # (128,3)
    ang_vel_mag = np.linalg.norm(ang_vel, axis=1)
    
    # Sliding Window Extraction
    for t in range(K-1, 128):  
        window = sample[t-K+1 : t+1, :]  # (K,10)
        
      
        buf = window
        feat = []
        feat.extend(buf[-1])                     
        feat.extend(np.mean(buf, axis=0))        
        feat.extend(np.std(buf, axis=0))         
        feat.extend(buf[-1] - buf[-2])           
        X_feat.append(feat)
        
        
        y_class.append(y_encoded[idx])
        group_feat.append(sid)
        
        
        #
        local_ang_vel = ang_vel_mag[t-K+1 : t+1]
        slope_dynamic = np.std(local_ang_vel) * 0.5   
        slope = base_slope + slope_dynamic
        slope = np.clip(slope, 0, 5)   
        
        
        local_acc_z = acc_z[t-K+1 : t+1]
       
        if len(local_acc_z) >= 3:
            b, a = signal.butter(5, 0.2, 'highpass')
            high_freq = signal.filtfilt(b, a, local_acc_z)
            roughness_dynamic = np.std(high_freq) * 2.0
        else:
            roughness_dynamic = 0
        roughness = base_rough + roughness_dynamic
        roughness = np.clip(roughness, 0, 1)
        
  
        
        dt = 1 / 100.0  #default imu_hz
        vel_z = np.cumsum(local_acc_z) * dt
        elev_change = vel_z[-1] - vel_z[0]   
      
        disp_z = np.cumsum(vel_z) * dt
        elev = disp_z[-1] - disp_z[0]   
        elev = np.clip(elev, -0.2, 0.2) + base_elev
        y_elev.append(elev)
        
        y_slope.append(slope)
        y_rough.append(roughness)


X_feat = np.array(X_feat, dtype=np.float32)
y_class = np.array(y_class)
y_slope = np.array(y_slope, dtype=np.float32)
y_rough = np.array(y_rough, dtype=np.float32)
y_elev = np.array(y_elev, dtype=np.float32)
group_feat = np.array(group_feat)


feature_cols = [f'feat_{i}' for i in range(40)]  
df_out = pd.DataFrame(X_feat, columns=feature_cols)
df_out['label_class'] = y_class
df_out['slope'] = y_slope
df_out['roughness'] = y_rough
df_out['elevation_change'] = y_elev
df_out['series_id'] = group_feat
df_out.to_csv('./Modified_Datasets/frame_multi_target.csv', index=False)

#This file for datasets procession --Maintainer: Jiming Yang
