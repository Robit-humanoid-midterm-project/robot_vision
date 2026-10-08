"""Fit observed row centers of each class in ORIGINAL image coordinates."""
import cv2
import numpy as np


def fit_lanes(result, shape, min_height_fraction=0.15, max_fit_error=8.0):
    height, width = shape[:2]
    selected = {}
    union = np.zeros((height, width), np.uint8)
    if result.masks is None:
        return selected, union
    for polygon, box in zip(result.masks.xy, result.boxes):
        side = {"left-sideline": "left", "right-sideline": "right"}.get(result.names[int(box.cls.item())])
        if side is None or len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        points = np.rint(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [points], 255)
        # Equal weight per observed row prevents a wide/near stripe dominating the fit.
        ys = np.flatnonzero(np.any(mask, axis=1))
        if len(ys) < max(3, int(height * min_height_fraction)):
            continue
        xs = np.array([np.median(np.flatnonzero(mask[y])) for y in ys])
        y_center = float(np.mean(ys))
        slope, intercept = np.linalg.lstsq(
            np.column_stack([ys - y_center, np.ones(len(ys))]), xs, rcond=None)[0]
        predicted = slope * (ys - y_center) + intercept
        error = float(np.sqrt(np.mean((xs - predicted) ** 2)))
        if not np.isfinite(error) or error > max_fit_error:
            continue
        ymin, ymax = int(ys[0]), int(ys[-1])
        x_at = lambda y: float(slope * (y - y_center) + intercept)
        # The reference remains within the observed stripe; no extension to the image bottom.
        reference = float(np.clip(0.85 * (height - 1), ymin, ymax))
        confidence = float(box.conf.item())
        line = dict(side=side, top=(x_at(ymin), float(ymin)), bottom=(x_at(ymax), float(ymax)),
                    observed_y_min=ymin, observed_y_max=ymax, reference_y_px=reference,
                    line_x_at_reference_px=x_at(reference),
                    pixel_separation_px=abs(x_at(reference) - (width - 1) * 0.5),
                    fit_error_px=error, confidence=confidence)
        # One candidate per side. Highest confidence wins; visible height breaks ties.
        rank = (confidence, len(ys))
        if side not in selected or rank > selected[side][0]:
            selected[side] = (rank, line, mask)
    lines = {}
    for side, (_, line, mask) in selected.items():
        lines[side] = line
        union[mask > 0] = 255
    return lines, union
