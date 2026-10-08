#!/usr/bin/env python3
"""Extract evenly spaced raw camera frames and keep their source timestamps."""

import argparse
import csv
from pathlib import Path

import cv2


def field_indices(capture: cv2.VideoCapture, frame_count: int) -> list[int]:
    """Find the initial field run and reject frames with little visible turf."""
    samples = []
    for index in range(0, frame_count, 10):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot inspect frame {index}")
        hsv = cv2.cvtColor(frame[180:450, 60:580], cv2.COLOR_BGR2HSV)
        turf_fraction = cv2.countNonZero(cv2.inRange(hsv, (30, 25, 20),
                                                     (95, 255, 255))) / hsv.shape[0] / hsv.shape[1]
        samples.append((index, turf_fraction))
    cutoff = frame_count
    for position in range(len(samples) - 2):
        if all(score < 0.2 for _, score in samples[position:position + 3]):
            cutoff = samples[position][0]
            break
    return [index for index, score in samples if index < cutoff and score >= 0.35]


def timestamps_for(video: Path) -> list[int]:
    path = Path(f"{video}.timestamps.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    stamps = [int(row["ros_time_ns"]) for row in rows]
    if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"Invalid or empty frame timestamps: {path}")
    return stamps


def select_indices(stamps: list[int], count: int) -> list[int]:
    """Pick time-quantiles without duplicating frame indices."""
    if count >= len(stamps):
        return list(range(len(stamps)))
    start, end = stamps[0], stamps[-1]
    selected = set()
    for slot in range(count):
        target = start + (end - start) * (slot + 0.5) / count
        index = min(range(len(stamps)), key=lambda i: abs(stamps[i] - target))
        selected.add(index)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=24)
    parser.add_argument("--field-only", action="store_true",
                        help="Sample only the first continuous section showing the green field")
    parser.add_argument("--videos", nargs="*", help="Video stems without .avi; default: all")
    args = parser.parse_args()
    if args.frames_per_video < 1:
        parser.error("--frames-per-video must be positive")
    videos = ([args.input_dir / f"{stem}.avi" for stem in args.videos]
              if args.videos else sorted(args.input_dir.glob("*.avi")))
    if not videos or any(not video.is_file() for video in videos):
        parser.error("One or more videos are missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty to avoid mixing extraction runs")
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    total = 0
    with manifest_path.open("w", newline="") as stream:
        manifest = csv.writer(stream)
        manifest.writerow(["image", "video", "frame_index", "ros_time_ns", "seconds_from_start"])
        for video in videos:
            stamps = timestamps_for(video)
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(stamps):
                raise ValueError(f"Video/CSV frame count mismatch: {video}")
            candidates = (field_indices(capture, len(stamps)) if args.field_only
                          else list(range(len(stamps))))
            if len(candidates) < args.frames_per_video:
                raise ValueError(f"Only {len(candidates)} field frames found: {video}")
            selected = [candidates[i] for i in select_indices(
                [stamps[index] for index in candidates], args.frames_per_video)]
            saved = 0
            for index in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame.shape[:2] != (480, 640):
                    raise ValueError(f"Cannot read calibrated frame {index}: {video}")
                filename = f"{video.stem}_f{index:06d}.jpg"
                if not cv2.imwrite(str(images_dir / filename), frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Cannot write extracted image: {filename}")
                manifest.writerow([filename, video.name, index, stamps[index],
                                   f"{(stamps[index] - stamps[0]) / 1e9:.3f}"])
                saved += 1
            capture.release()
            total += saved
            print(f"{video.name}: {saved} frames")
    print(f"Saved {total} images to {images_dir}")
    print(f"Source mapping: {manifest_path}")


if __name__ == "__main__":
    main()
