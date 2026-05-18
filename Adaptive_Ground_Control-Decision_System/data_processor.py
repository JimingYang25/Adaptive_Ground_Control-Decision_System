import pandas as pd
import numpy as np
from scipy import signal

# ------------------------------
# 1. 读取原始数据
# ------------------------------
X = pd.read_csv('./Dataset/X_train.csv')
y = pd.read_csv('./Dataset/y_train.csv')
X = X.sort_values(['series_id', 'measurement_number'])

feature_cols = [c for c in X.columns if c not in ['row_id', 'series_id', 'measurement_number']]

# ------------------------------
# 2. 构建时序数据 (samples, 128, 10)
# ------------------------------
X_seq = []
series_ids = []
for sid, group in X.groupby('series_id'):
    mat = group[feature_cols].values
    if mat.shape[0] == 128:
        X_seq.append(mat)
        series_ids.append(sid)
X_seq = np.array(X_seq, dtype=np.float32)  # (n_segments, 128, 10)

# 标签映射
y_filtered = y[y['series_id'].isin(series_ids)].drop_duplicates('series_id')
y_aligned = y_filtered.set_index('series_id').loc[series_ids]
surface_names = y_aligned['surface'].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(surface_names)
num_classes = len(le.classes_)

# 定义类别到基准值的映射（经验值）
# 坡度(度), 粗糙度(0~1), 高程变化(m/s)
surface_base = {
    'carpet': (0.0, 0.3, 0.0),
    'concrete': (0.0, 0.1, 0.0),
    'fine_concrete': (0.0, 0.05, 0.0),
    'soft_tiles': (0.0, 0.35, 0.0),
    'tiled': (0.0, 0.25, 0.0),
    'soft_pvc': (0.0, 0.2, 0.0),
    'hard_tiles_large_space': (0.0, 0.15, 0.0),
    'grass': (1.5, 0.5, 0.02),      # 示例：假设有草地
    'gravel': (2.0, 0.8, 0.05),
    # 请根据实际 surface 名称补充完整
}

# 辅助函数：四元数转欧拉角（俯仰角）
def quat_to_pitch(qw, qx, qy, qz):
    # 返回俯仰角（弧度）
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    return np.arcsin(sinp)

# ------------------------------
# 3. 生成帧级样本（含连续标签）
# ------------------------------
K = 30   # 历史窗口长度
window_stride = 1  # 步长，可调

X_feat = []   # 40维特征
y_class = []  # 类别标签
y_slope = []  # 坡度
y_rough = []  # 粗糙度
y_elev = []   # 高程变化
group_feat = []

for idx, (sid, sample) in enumerate(zip(series_ids, X_seq)):
    surface = surface_names[idx]
    base_slope, base_rough, base_elev = surface_base.get(surface, (0.0, 0.2, 0.0))
    
    # 预计算一些序列特征用于动态调整
    # 提取z轴线性加速度（第9列，索引8）
    acc_z = sample[:, 8]   # (128,)
    # 计算角速度幅值（用于坡度动态）
    ang_vel = sample[:, 4:7]  # (128,3)
    ang_vel_mag = np.linalg.norm(ang_vel, axis=1)
    
    # 滑动窗口提取
    for t in range(K-1, 128):   # t 是当前帧索引（0-based）
        window = sample[t-K+1 : t+1, :]  # (K,10)
        
        # ----- 提取40维特征（同之前）-----
        buf = window
        feat = []
        feat.extend(buf[-1])                     # 当前帧
        feat.extend(np.mean(buf, axis=0))        # 均值
        feat.extend(np.std(buf, axis=0))         # 标准差
        feat.extend(buf[-1] - buf[-2])           # 差分
        X_feat.append(feat)
        
        # 类别标签
        y_class.append(y_encoded[idx])
        group_feat.append(sid)
        
        # ----- 合成连续标签 -----
        # 坡度：基准 + 窗口内角速度幅值的变化率（模拟上下坡）
        local_ang_vel = ang_vel_mag[t-K+1 : t+1]
        slope_dynamic = np.std(local_ang_vel) * 0.5   # 调节系数
        slope = base_slope + slope_dynamic
        slope = np.clip(slope, 0, 5)   # 限制范围 0~5度
        
        # 粗糙度：基准 + z轴加速度高频标准差（0~0.5额外）
        local_acc_z = acc_z[t-K+1 : t+1]
        # 高通滤波提取高频分量
        if len(local_acc_z) >= 3:
            b, a = signal.butter(5, 0.2, 'highpass')
            high_freq = signal.filtfilt(b, a, local_acc_z)
            roughness_dynamic = np.std(high_freq) * 2.0
        else:
            roughness_dynamic = 0
        roughness = base_rough + roughness_dynamic
        roughness = np.clip(roughness, 0, 1)
        
        # 高程变化：通过 z 轴速度积分（近似）窗口末帧与首帧的高度差
        # 先对加速度积分得到速度（假设初始速度为0），再积分得到高度变化
        dt = 1 / 100.0   # 假设IMU频率100Hz，每步0.01s
        vel_z = np.cumsum(local_acc_z) * dt
        elev_change = vel_z[-1] - vel_z[0]   # 末帧与首帧的速度差近似为高度变化率？
        # 更合理：位移 = 积分速度
        disp_z = np.cumsum(vel_z) * dt
        elev = disp_z[-1] - disp_z[0]   # 窗口内高度变化（米）
        elev = np.clip(elev, -0.2, 0.2) + base_elev
        y_elev.append(elev)
        
        y_slope.append(slope)
        y_rough.append(roughness)

# 转换为数组
X_feat = np.array(X_feat, dtype=np.float32)
y_class = np.array(y_class)
y_slope = np.array(y_slope, dtype=np.float32)
y_rough = np.array(y_rough, dtype=np.float32)
y_elev = np.array(y_elev, dtype=np.float32)
group_feat = np.array(group_feat)

# 保存为语义化CSV（包含特征 + 分类标签 + 连续标签）
feature_cols = [f'feat_{i}' for i in range(40)]  # 临时列名，也可用语义名
df_out = pd.DataFrame(X_feat, columns=feature_cols)
df_out['label_class'] = y_class
df_out['slope'] = y_slope
df_out['roughness'] = y_rough
df_out['elevation_change'] = y_elev
df_out['series_id'] = group_feat
df_out.to_csv('./Modified_Datasets/frame_multi_target.csv', index=False)
print("预处理完成，保存至 frame_multi_target.csv")