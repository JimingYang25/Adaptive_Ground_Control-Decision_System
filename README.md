# Adaptive_Ground_Control-Decision_System

# Description:

Based on model ( Vogue_Ming ) trained in pytorch framework,provide real-time ground/terrain info predictions and interfaces for control-decision adaptation

<pre>
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS 2 REAL‑TIME PIPELINE                          │
│                              (C++ / Python)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

   IMU sensor
   (100 Hz)
        │
        ▼
┌───────────────────┐
│  /imu/data_raw    │   (sensor_msgs/Imu)
│  subscriber       │
└─────────┬─────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Sliding Window Buffer                       │
│                     (deque, length = K = 30)                     │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  │
│   │ t-29│ t-28│ ... │ t-3 │ t-2 │ t-1 │ t   │     │     │     │  │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘  │
│   each element: 10‑dim IMU (quat + angular + linear)            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Feature Extractor (utils.py)                  │
│  Extract per‑window:                                             │
│   • current frame (10)                                           │
│   • mean of window (10)                                          │
│   • std of window (10)                                           │
│   • diff (curr - prev) (10)                                      │
│  → 40‑dim feature vector                                         │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     StandardScaler (fitted)                      │
│                (z = (x - μ) / σ, saved as scaler.pkl)            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Vogue_Ming Model                           │
│                   (Multi‑task MLP, PyTorch)                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Shared Backbone (MLP)                      ││
│  │   Linear(40→128) → BN → ReLU → Dropout(0.4)                 ││
│  │   Linear(128→64) → BN → ReLU → Dropout(0.4)                 ││
│  └───────────────────┬─────────────────────────────────────────┘│
│                      │                                           │
│          ┌───────────┼───────────┬───────────────┐              │
│          ▼           ▼           ▼               ▼              │
│   ┌────────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐       │
│   │Class Head  │ │Slope   │ │Roughness │ │Elevation     │       │
│   │Linear(64→C)│ │Head    │ │Head      │ │Change Head   │       │
│   │            │ │Linear→1│ │Linear→1  │ │Linear→1      │       │
│   └─────┬──────┘ └───┬────┘ └────┬─────┘ └──────┬───────┘       │
│         │            │           │              │               │
│    softmax          └───────────┴──────────────┘               │
│         │                    │                                  │
│    predicted          continuous predictions                    │
│    class index        (slope, roughness, elev_change)           │
└─────────┼────────────────────┼──────────────────────────────────┘
          │                    │
          ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LabelEncoder (inverse)                        │
│               (saved as label_encoder.pkl)                       │
│         convert class index → string (e.g. "carpet")             │
└─────────┬────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Publish /terrain_info                         │
│   Custom message (TerrainInfo.msg):                              │
│   • string surface                                               │
│   • float32 slope                                                │
│   • float32 roughness                                            │
│   • float32 elevation_change                                     │
│   • float32 confidence                                           │
└─────────┬────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│               Downstream consumers (control, planning)           │
│   • Adjust max speed based on slope/roughness                   │
│   • Switch control gains per surface type                       │
│   • Detect elevation obstacles                                   │
└──────────────────────────────────────────────────────────────────┘  
<pre/>
