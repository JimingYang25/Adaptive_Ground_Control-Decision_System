#!/usr/bin/env python3
# The system_node --Maintainer: Jiming Yang

import rclpy
from rclpy.node import Node
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Imu
from terrain_interfaces.msg import TerrainInfo

from terrain_classifier_pkg.Vogue_Ming import Vogue_Ming
from terrain_classifier_pkg.utilits import extract_features

class TerrainClassifierNode(Node):
    def __init__(self):
        super().__init__('terrain_classifier_node')

        self.declare_parameter('window_size', 30)
        self.declare_parameter('publish_freq_hz', 10.0)
        self.declare_parameter('imu_topic', '/imu/data_raw')
        self.declare_parameter('debug', False)

        self.window_size = self.get_parameter('window_size').value
        self.pub_period = 1.0 / self.get_parameter('publish_freq_hz').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.debug = self.get_parameter('debug').value

        pkg_share = get_package_share_directory('terrain_classifier_pkg')
        state_dict_path = os.path.join(pkg_share, 'models', 'multi_task_best.pth')
        scaler_path = os.path.join(pkg_share, 'models', 'scaler.pkl')
        le_path = os.path.join(pkg_share, 'models', 'label_encoder.pkl')

        self.scaler = joblib.load(scaler_path)
        self.le = joblib.load(le_path)
        num_classes = len(self.le.classes_)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        self.model = Vogue_Ming()
        try:
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.get_logger().info("Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            rclpy.shutdown()
            return
        self.model.to(self.device)
        self.model.eval()

        # Window
        self.buffer = deque(maxlen=self.window_size)

        self.sub = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)
        self.pub = self.create_publisher(TerrainInfo, '/terrain_info', 10)

        self.last_pub = self.get_clock().now()
        self.get_logger().info('Terrain classifier node activated')

    def imu_callback(self, msg):
        q = msg.orientation
        av = msg.angular_velocity
        la = msg.linear_acceleration
        raw = [q.w, q.x, q.y, q.z, av.x, av.y, av.z, la.x, la.y, la.z]
        self.buffer.append(raw)

        if len(self.buffer) < self.window_size:
            return
        
        now = self.get_clock().now()
        if (now - self.last_pub).seconds < self.pub_period:
            return
        self.last_pub = now

        feat = extract_features(list(self.buffer)).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)

        with torch.no_grad():
            input_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(self.device)
            cls_logits, slope, rough, elev = self.model(input_tensor)
            pred = torch.argmax(cls_logits, dim=1).item()
            surface = self.le.inverse_transform([pred])[0]
            confidence = torch.softmax(cls_logits, dim=1).max().item()

        msg_out = TerrainInfo()
        msg_out.surface = surface
        msg_out.slope = float(slope.item())
        msg_out.roughness = float(rough.item())
        msg_out.elevation_change = float(elev.item())
        msg_out.confidence = confidence
        self.pub.publish(msg_out)

        if self.debug:
            self.get_logger().debug(f"Pred: {surface}, slope={msg_out.slope:.2f}, rough={msg_out.roughness:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = TerrainClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# The system_node --Maintainer: Jiming Yang
