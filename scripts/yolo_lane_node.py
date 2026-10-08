#!/usr/bin/env python3
"""YOLO segmentation worker: latest camera frame in, synchronized LaneFrame out."""
import sys
import threading
import time
from pathlib import Path
from yolo_runtime_paths import default_weights

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from robot_vision.msg import LaneFrame, LaneLine
from yolo_lane_fit import fit_lanes


class YoloLaneNode(Node):
    def __init__(self):
        super().__init__("yolo_lane")
        def param(name, value):
            return self.declare_parameter(name, value).value
        self.weights = str(Path(param("weights", default_weights())).expanduser())
        self.topic = param("image_topic", "/camera1/camera/compressed_image")
        self.output = param("output_topic", "/vision/yolo_lane_frame")
        self.conf = param("confidence", 0.5)
        self.imgsz = param("image_size", 640)
        self.device = param("device", "cpu")
        self.fps = param("max_processing_fps", 15.0)
        self.max_age = param("max_frame_age_s", 1.0)
        self.min_height = param("min_observed_height_fraction", 0.15)
        self.max_error = param("max_fit_error_px", 8.0)
        if not (0 < self.conf <= 1 and self.fps > 0 and self.max_age > 0 and
                0 < self.min_height <= 1 and self.max_error > 0 and self.imgsz >= 32):
            raise ValueError("Invalid YOLO lane parameters")
        if not Path(self.weights).is_file():
            raise FileNotFoundError(self.weights)
        from ultralytics import YOLO
        self.model = YOLO(self.weights)
        if self.model.task != "segment" or not {"left-sideline", "right-sideline"}.issubset(self.model.names.values()):
            raise ValueError(f"Wrong model task/classes: {self.model.task}, {self.model.names}")
        # Warm up once before accepting camera frames.
        self.model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz,
                           conf=self.conf, device=self.device, rect=False, verbose=False)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(LaneFrame, self.output, qos)
        self.lock = threading.Lock()
        self.latest = None
        self.subscription = self.create_subscription(CompressedImage, self.topic, self.receive, qos)
        self.get_logger().info(f"YOLO ready: {self.weights}; {self.topic} -> {self.output}; classes={self.model.names}")

    def receive(self, message):
        with self.lock:
            self.latest = (message, time.monotonic())

    def process(self):
        with self.lock:
            pending, self.latest = self.latest, None
        if pending is None:
            return
        image, received = pending
        if time.monotonic() - received > self.max_age:
            return
        output = LaneFrame()
        output.image = image
        output.left.header = output.right.header = output.mask.header = image.header
        started = time.monotonic()
        try:
            frame = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid camera JPEG")
            result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                        retina_masks=True, rect=False, verbose=False)[0]
            lines, mask = fit_lanes(result, frame.shape, self.min_height, self.max_error)
            for side, values in lines.items():
                line = LaneLine()
                line.header = image.header
                line.valid = True
                line.side = side
                line.top.x, line.top.y = values.pop("top")
                line.bottom.x, line.bottom.y = values.pop("bottom")
                for key, value in values.items():
                    setattr(line, key, value)
                setattr(output, side, line)
            success, encoded = cv2.imencode(".png", mask)
            if not success:
                raise ValueError("Cannot encode lane mask")
            output.mask.format = "png"
            output.mask.data = encoded.tobytes()
            output.status = "ok"
        except Exception as error:
            # Invalid outputs never reuse detections from an earlier frame.
            output.left = LaneLine(header=image.header)
            output.right = LaneLine(header=image.header)
            output.status = f"inference_error: {error}"
            self.get_logger().error(output.status)
        output.inference_ms = float((time.monotonic() - started) * 1000)
        if time.monotonic() - received <= self.max_age:
            self.publisher.publish(output)


def main():
    rclpy.init()
    node = None
    thread = None
    try:
        node = YoloLaneNode()
        def spin():
            try:
                rclpy.spin(node)
            except ExternalShutdownException:
                pass
        thread = threading.Thread(target=spin, daemon=True)
        thread.start()
        while rclpy.ok():
            started = time.monotonic()
            node.process()
            time.sleep(max(0, 1.0 / node.fps - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"YOLO lane startup failed: {error}", file=sys.stderr)
        return 1
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if thread:
            thread.join(timeout=2)
        if node:
            node.destroy_node()
    return 0


if __name__ == "__main__":
    sys.exit(main())
