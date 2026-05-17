# Adaptive_Ground_Control-Decision_System

   ## Description:

Based on model ( Vogue_Ming ) trained in pytorch framework,provide real-time ground/terrain info predictions and interfaces for control-decision adaptation

   ## Architecture:
<pre>
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS 2 REAL‑TIME PIPELINE                          │
│                              (C++ / Python)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

   IMU sensor → orientation (w,x,y,z) / angular_velocity (x,y,z) / linear_acceleration (x,y,z) → 10 features
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
│                    Feature Extractor (📄utils.py)                 │
│  Extract per‑window:                                             │
│   • current frame (10)                                           │
│   • mean of window (10)                                          │
│   • std of window (10)                                           │
│   • diff (curr - prev) (10)                                      │
│  → **40‑dim feature vector**                                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     StandardScaler (fitted)                      │
│                (z = (x - μ) / σ, saved as scaler.pkl)            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Vogue_Ming Model                                 │
│                          (Multi‑task, PyTorch)                                │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           40‑dim input                                 │  │
│  │                                  │                                      │  │
│  │                    ┌─────────────┴─────────────┐                        │  │
│  │                    ▼                           ▼                        │  │
│  │           [0:10] (current frame)      [10:40] (window stats)            │  │
│  │                    │                           │                        │  │
│  │                    ▼                           ▼                        │  │
│  │          frame_branch                 window_branch                     │  │
│  │         Linear(10→60)                Linear(30→180)                     │  │
│  │         BN, ReLU, Dropout            BN, ReLU, Dropout                  │  │
│  │                    │                           │                        │  │
│  │                    └───────────┬───────────────┘                        │  │
│  │                                ▼                                        │  │
│  │                    Concatenate → 240‑dim                                │  │
│  │                                │                                        │  │
│  │                                ▼                                        │  │
│  │                    Shared Backbone (MLP)                                │  │
│  │          Linear(240→480) → BN → ReLU → Dropout                          │  │
│  │          Linear(480→240) → BN → ReLU → Dropout                          │  │
│  │          Linear(240→120) → BN → ReLU → Dropout                          │  │
│  │          Linear(120→40)  → BN → ReLU → Dropout                          │  │
│  │                                │                                        │  │
│  │          ┌───────────┬────────┼───────────┬────────────┐               │  │
│  │          ▼           ▼        ▼           ▼            ▼               │  │
│  │   ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐     │  │
│  │   │Class Head│ │Slope    │ │Roughness│ │Elevation│ │ Confidence  │     │  │
│  │   │Linear(40→│ │Head     │ │Head     │ │Head     │ │(softmax max)│     │  │
│  │   │C)        │ │Linear→1 │ │Linear→1 │ │Linear→1 │ │             │     │  │
│  │   └────┬─────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘     │  │
│  │        │            │          │           │              │            │  │
│  │    softmax          └──────────┴───────────┘              │            │  │
│  │        │                        │                         │            │  │
│  │   class index           continuous predictions            │            │  │
│  │   (0..C-1)            (slope, roughness, elev)            │            │  │
│  └────────┼────────────────────────┼─────────────────────────┼────────────┘  │
│           │                        │                         │               │
└───────────┼────────────────────────┼─────────────────────────┼───────────────┘
            │                        │                         │
            ▼                        ▼                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LabelEncoder (inverse)                                    │
│               (saved as label_encoder.pkl)                                   │
│         convert class index → string (e.g. "carpet")                         │
└─────────┬────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Publish /terrain_info                                     │
│   Custom message (TerrainInfo.msg):                                          │
│   • string surface                                                           │
│   • float32 slope                                                            │
│   • float32 roughness                                                        │
│   • float32 elevation_change                                                 │
│   • float32 confidence                                                       │
└─────────┬────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│               Downstream consumers (control, planning)                       │
│   • Adjust max speed based on slope/roughness                               │
│   • Switch control gains per surface type                                   │
│   • Detect elevation obstacles                                              │
└──────────────────────────────────────────────────────────────────────────────┘
</pre>

   ### Model - Vogue_Ming:

Original datasets from :

   @misc{career-con-2019,
    author = {Maggie and Sohier Dane},
    title = {CareerCon 2019 - Help Navigate Robots },
    year = {2019},
    howpublished = {\url{https://kaggle.com/competitions/career-con-2019}},
    note = {Kaggle}
}
<pre>
🔧Model trained by reconstracted dataset: 📄
</pre>
<pre>
<img width="2100" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/19b4f963-0fbe-40e0-b6bd-90b3ba84b4cb" />

</pre>


























