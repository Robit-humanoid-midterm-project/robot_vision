#!/usr/bin/env python3
"""Preview a trained segmentation model on the Insta360 ROS image stream."""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

COLORS = {"left-sideline": (255, 180, 0), "right-sideline": (0, 220, 255)}


def render(frame, result):
    overlay = frame.copy()
    detections = []
    if result.masks is None:
        return overlay, detections
    for polygon, box in zip(result.masks.xy, result.boxes):
        name = result.names[int(box.cls.item())]
        if name not in COLORS or len(polygon) < 3:
            continue
        points = np.rint(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, frame.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, frame.shape[0] - 1)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [points], 255)
        color = COLORS[name]
        overlay[mask > 0] = (overlay[mask > 0] * 0.65 + np.array(color) * 0.35).astype(np.uint8)
        cv2.polylines(overlay, [points], True, color, 2)
        # Row midpoints describe the observed stripe without extrapolating hidden parts.
        centers = []
        for y in range(int(points[:, 1].min()), int(points[:, 1].max()) + 1, 4):
            xs = np.flatnonzero(mask[y])
            if len(xs):
                centers.append((int(np.median(xs)), y))
        if len(centers) > 1:
            cv2.polylines(overlay, [np.array(centers, np.int32)], False, color, 2)
        confidence = float(box.conf.item())
        anchor = tuple(points[np.argmin(points[:, 1])])
        cv2.putText(overlay, f"{name} {confidence:.2f}",
                    (int(anchor[0]), max(25, int(anchor[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        detections.append({"class": name, "confidence": confidence,
                           "polygon": points.tolist(), "centerline": centers})
    return overlay, detections


def predict(model, frame, args):
    return model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                         device=args.device, retina_masks=True, rect=False,
                         verbose=False)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="/home/doyeon/Downloads/weights.pt")
    parser.add_argument("--topic", default="/camera1/camera/compressed_image")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image", help="Test a saved image instead of ROS camera")
    parser.add_argument("--output", default="line_preview_result.jpg")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or args.timeout <= 0 or args.imgsz < 32:
        parser.error("conf must be (0,1], timeout positive, imgsz >= 32")
    if not Path(args.weights).is_file():
        parser.error(f"Weights not found: {args.weights}")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    if model.task != "segment" or not set(COLORS).issubset(set(model.names.values())):
        raise RuntimeError(f"Expected left-sideline/right-sideline segmentation; got {model.task}: {model.names}")
    print(f"Loaded classes: {model.names}", flush=True)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {args.image}")
        result = predict(model, frame, args)
        overlay, detections = render(frame, result)
        if not cv2.imwrite(args.output, np.hstack([frame, overlay])):
            raise RuntimeError(f"Cannot write {args.output}")
        Path(args.output).with_suffix(".json").write_text(json.dumps(detections, indent=2))
        print(f"Saved {args.output}; {len(detections)} boundaries; {result.speed}")
        return
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    rclpy.init()
    node = Node("insta360_yolo_line_preview")
    lock = threading.Lock()
    latest = {"frame": None, "time": 0.0, "sequence": 0}

    def receive(message):
        frame = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            with lock:
                latest.update(frame=frame, time=time.monotonic(), sequence=latest["sequence"] + 1)

    subscription = node.create_subscription(CompressedImage, args.topic, receive, qos_profile_sensor_data)
    worker = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    worker.start()
    window = "Insta360 | Original + YOLO boundaries (Q/Esc: quit, S: save)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 540)
    sequence = -1
    panel = np.zeros((480, 1280, 3), np.uint8)
    print(f"Waiting for {args.topic}. Q/Esc quits; S saves a snapshot.", flush=True)
    try:
        while rclpy.ok():
            with lock:
                frame, received, current = latest["frame"], latest["time"], latest["sequence"]
            if frame is None or time.monotonic() - received > args.timeout:
                panel = np.zeros((480, 1280, 3), np.uint8)
                cv2.putText(panel, "Waiting for Insta360 camera stream...", (25, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            elif current != sequence:
                started = time.monotonic()
                result = predict(model, frame, args)
                overlay, detections = render(frame, result)
                elapsed = time.monotonic() - started
                panel = np.hstack([frame, overlay])
                cv2.putText(panel, f"{len(detections)} boundaries | {elapsed*1000:.0f} ms | conf {args.conf:.2f}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                sequence = current
            cv2.imshow(window, panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                path = f"line_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(path, panel)
                print(f"Saved {path}", flush=True)
    finally:
        rclpy.shutdown()
        worker.join(timeout=2)
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
