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
│                     (deque, length = K )                         │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  │
│   │ t-K │t-K+1│ ... │ t-3 │ t-2 │ t-1 │ t   │     │     │     │  │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘  │
│   each element: 10‑dim IMU (quat + angular + linear)             │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Feature Extractor                             |
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
│                              Vogue_Ming Model                                │
│                          (Multi‑task, PyTorch)                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           40‑dim input                                 │  │
│  │                                  │                                     │  │
│  │                    ┌─────────────┴─────────────┐                       │  │
│  │                    ▼                           ▼                       │  │
│  │           [0:10] (current frame)      [10:40] (window stats)           │  │
│  │                    │                           │                       │  │
│  │                    ▼                           ▼                       │  │
│  │          frame_branch                 window_branch                    │  │
│  │         Linear(10→80)                Linear(30→240)                    │  │
│  │         BN, ReLU, Dropout            BN, ReLU, Dropout                 │  │
│  │                    │                           │                       │  │
│  │                    └───────────┬───────────────┘                       │  │
│  │                                ▼                                       │  │
│  │                    Concatenate → 320‑dim                               │  │
│  │                                │                                       │  │
│  │                                ▼                                       │  │
│  │                          Shared Backbone                               │  │
│  │             320 → Linear(240→40) → BN → ReLU → Dropout                 │  │
│  │                               │                                        │  │
│  │          ┌───────────┬────────┼───────────┬────────────┐               │  │
│  │          ▼           ▼        ▼           ▼            ▼               │  │
│  │   ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐     │  │
│  │   │Class Head│ │Slope    │ │Roughness│ │Elevation│ │ Confidence  │     │  │
│  │   │Linear(40→│ │Head     │ │Head     │ │Head     │ │(softmax max)│     │  │
│  │   │C)        │ │Linear→1 │ │Linear→1 │ │Linear→1 │ │             │     │  │
│  │   └────┬─────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘     │  │
│  │        │            │           │           │             │            │  │
│  │    softmax          └───────────┴───────────┘             │            │  │
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
│               Downstream consumers (Other Framework / Simulation)            │
│   • Adjust max speed based on slope/roughness                                │
│   • Switch control gains per surface type                                    │
│   • Detect elevation obstacles                                               │
└──────────────────────────────────────────────────────────────────────────────┘
</pre>

   ### Model - Vogue_Ming:

#### Original datasets from :

   @misc{career-con-2019,
    author = {Maggie and Sohier Dane},
    title = {CareerCon 2019 - Help Navigate Robots },
    year = {2019},
    howpublished = {\url{https://kaggle.com/competitions/career-con-2019}},
    note = {Kaggle}
}

#### 🔧Model trained by reconstracted dataset: 📄[frame_multi_target.csv](https://github.com/JimingYang25/Adaptive_Ground_Control-Decision_System/blob/main/frame_multi_target.csv) in method 📄[data_processor.py](https://github.com/JimingYang25/Adaptive_Ground_Control-Decision_System/blob/main/data_processor.py)


<pre>
Latest training log:
   Totola epoch : 109
   
   Optimal Save Point (Epoch : 100)

   Optimal Validation Accauracy:0.9294（92.94%）

   Relevent Loss：Train Loss: 0.6647 | Val Loss: 0.6656 | Val Acc: 0.9294

   Early Stopping Enabled at Epoch 109 (non-decreasing Validation loss for 10 Epoch )
</pre>
<pre>
<img width="2100" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/ea6bb2c6-630a-49f7-884e-5ec6f22c1d3d" />


</pre>

## Quick Start🔧


## 1. Create a ROS2 workspace / Enter your workplaces

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## 2. Clone the branch

```bash
cd ~/ros2_ws
git clone -b Jazzy https://github.com/JimingYang25/Adaptive_Ground_Control-Decision_System.git
```

## 3. Build the workspace

```bash
cd ~/ros2_ws
colcon build --packages-select terrain_interfaces terrain_classifier_pkg
source install/setup.bash
```

## Configuration

Edit the parameter file `src/terrain_classifier_pkg/config/terrain_params.yaml`:

```yaml
/**:
  ros__parameters:
    window_size: 30          # Size of sliding buffer (K frames)
    publish_freq_hz: 10.0    # Maximum publishing rate (Hz)
    imu_topic: "/imu/data_raw" # imu_msg topic 
    debug: false  # show debug info or not
```

You can also override parameters at launch time (see below).

## Running the Node

### Basic launch

```bash
ros2 launch terrain_classifier_pkg terrain_classifier.launch.py
```

### Override parameters from command line

```bash
ros2 launch terrain_classifier_pkg terrain_classifier.launch.py \
    window_size:=40 \
    publish_freq_hz:=5.0 \
    imu_topic:=/my_imu/data
```

### Run node directly (without launch file)

```bash
ros2 run terrain_classifier_pkg terrain_node.py \
    --ros-args \
    -p window_size:=30 \
    -p publish_freq_hz:=10.0 \
    -p imu_topic:=/imu/data_raw
```

## Output

The node publishes to `/terrain_info` using the custom `TerrainInfo` message.

### View the output

```bash
ros2 topic echo /terrain_info
```

Example:

```
surface: "carpet"
slope: 0.12
roughness: 0.34
elevation_change: 0.01
confidence: 0.98
```

## Testing with a ROS2 bag

```bash
ros2 bag play /path/to/imu_data.bag
ros2 topic echo /terrain_info
```

Or publish dummy IMU data:

```bash
ros2 topic pub /imu/data_raw sensor_msgs/msg/Imu "{...}"
```


## Adding C++ Components

The package uses `ament_cmake` – available to add C++ components.  
Place your `.cpp` files in `src/` and extend `CMakeLists.txt`.

---

## License

Apache 2.0

---

## Repository Links

- [terrain_interfaces](https://github.com/JimingYang25/Adaptive_Ground_Control-Decision_System)
- [terrain_classifier_pkg](https://github.com/JimingYang25/Adaptive_Ground_Control-Decision_System)

---

## Citation
If you use this project in your research or academic work, please cite our repository as follows:

```bibtex
@misc{yang2026adaptivegroundcontrol,
  title={Adaptive Ground Control and Decision System Based on ROS2},
  author={Jiming Yang},
  year={2026},
  publisher={GitHub},
  journal={GitHub Repository},
  url={https://github.com/JimingYang25/Adaptive_Ground_Control-Decision_System},
  note={Published: 2026-05-18}
}
```




















