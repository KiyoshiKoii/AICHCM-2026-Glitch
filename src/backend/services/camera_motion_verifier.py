"""Lightweight runtime verifier for explicit camera-motion constraints.

It intentionally complements Qwen/KIS instead of replacing it: KIS identifies
the subject/objects in a candidate frame, then this verifier checks only the
camera/shot properties that are not observable from one still image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAMERA_CUE_RE = re.compile(
    r"(?:máy\s+quay|góc\s+quay|lia\s+máy|pan|tilt|zoom|"
    r"chuyển\s+(?:cảnh|sang)|cú\s+máy|shot|tĩnh|đứng\s+yên)",
    flags=re.IGNORECASE,
)
UP_CUE_RE = re.compile(r"(?:lia|quay|máy\s+quay|tilt|pan|chéo).{0,28}\blên\b", re.IGNORECASE)
DOWN_CUE_RE = re.compile(r"(?:lia|quay|máy\s+quay|tilt|pan|chéo).{0,28}\bxuống\b", re.IGNORECASE)
STATIC_CUE_RE = re.compile(r"(?:cú\s+máy\s+tĩnh|góc\s+tĩnh|đứng\s+yên|cố\s+định)", re.IGNORECASE)
TRANSITION_CUE_RE = re.compile(r"(?:chuyển\s+cảnh|chuyển\s+sang|cắt\s+cảnh|shot)", re.IGNORECASE)
DIAGONAL_CUE_RE = re.compile(r"(?:chéo|diagonal)", re.IGNORECASE)


@dataclass(frozen=True)
class CameraMotionResult:
    score: float
    reason: str
    details: dict[str, Any]


class CameraMotionVerifier:
    """Verify a small number of KIS candidate windows with OpenCV."""

    def __init__(self, video_dir: Path = Path("data/videos"), sample_fps: float = 5.0) -> None:
        self.video_dir = video_dir
        self.sample_fps = sample_fps

    @staticmethod
    def has_explicit_camera_constraint(description: str) -> bool:
        return bool(CAMERA_CUE_RE.search(description))

    @staticmethod
    def _constraints(description: str) -> dict[str, bool]:
        return {
            "up": bool(UP_CUE_RE.search(description)),
            "down": bool(DOWN_CUE_RE.search(description)),
            "static": bool(STATIC_CUE_RE.search(description)),
            "transition": bool(TRANSITION_CUE_RE.search(description)),
            "diagonal": bool(DIAGONAL_CUE_RE.search(description)),
        }

    @staticmethod
    def _score_metrics(metrics: dict[str, float], constraints: dict[str, bool]) -> float:
        """Map normalized optical-flow/shot evidence to one comparable score."""

        components: list[float] = []
        if constraints["up"] or constraints["down"]:
            expected = metrics["up_direction"] if constraints["up"] else metrics["down_direction"]
            direction_score = max(0.0, min(1.0, 0.5 + expected / 2.0))
            if constraints["diagonal"]:
                direction_score = 0.7 * direction_score + 0.3 * metrics["diagonal_fraction"]
            components.append(direction_score)
        if constraints["static"]:
            components.append(metrics["static_fraction"])
        if constraints["transition"]:
            components.append(metrics["transition_score"])
        return sum(components) / len(components) if components else 0.5

    def verify(
        self,
        *,
        video_id: str,
        description: str,
        frame_indices: list[int],
    ) -> CameraMotionResult:
        constraints = self._constraints(description)
        if not any(constraints.values()):
            return CameraMotionResult(
                score=0.5,
                reason="No explicit camera-motion constraint in this event.",
                details={"constraints": constraints, "available": False},
            )
        try:
            import cv2
            import numpy as np
        except ImportError:
            return CameraMotionResult(
                score=0.5,
                reason="OpenCV is unavailable; camera verification was skipped.",
                details={"constraints": constraints, "available": False},
            )

        video_path = self.video_dir / f"{video_id.upper()}.mp4"
        if not video_path.is_file():
            return CameraMotionResult(
                score=0.5,
                reason="Video file is unavailable; camera verification was skipped.",
                details={"constraints": constraints, "available": False},
            )

        results = [
            self._verify_anchor(cv2, np, video_path, max(0, frame_index), constraints)
            for frame_index in list(dict.fromkeys(frame_indices))[:3]
        ]
        available = [item for item in results if item is not None]
        if not available:
            return CameraMotionResult(
                score=0.5,
                reason="Could not decode a motion window around the KIS candidates.",
                details={"constraints": constraints, "available": False},
            )
        best = max(available, key=lambda item: item[0])
        return CameraMotionResult(
            score=best[0],
            reason="Camera/shot evidence verified around the strongest KIS candidate.",
            details={"constraints": constraints, "available": True, **best[1]},
        )

    def _verify_anchor(self, cv2, np, video_path: Path, frame_index: int, constraints: dict[str, bool]):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        anchor_seconds = frame_index / fps
        start_seconds = max(0.0, anchor_seconds - 10.0)
        end_seconds = anchor_seconds + 4.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
        step = max(1, round(fps / self.sample_fps))
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        previous: tuple[Any, Any] | None = None
        flows: list[tuple[float, float, float, float]] = []
        transitions: list[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_number / fps
            frame_number += 1
            if timestamp > end_seconds:
                break
            if frame_number % step:
                continue
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
            cv2.normalize(histogram, histogram)
            if previous is not None:
                difference = float(cv2.absdiff(gray, previous[0]).mean())
                correlation = float(cv2.compareHist(histogram, previous[1], cv2.HISTCMP_CORREL))
                change_score = difference / 35.0 + max(0.0, 1.0 - correlation)
                if change_score > 1.8 and (not transitions or timestamp - transitions[-1] > 0.8):
                    transitions.append(timestamp)
                points = cv2.goodFeaturesToTrack(previous[0], 250, 0.01, 6)
                if points is not None:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous[0], gray, points, None)
                    if next_points is not None and status is not None:
                        old = points[status.ravel() == 1].reshape(-1, 2)
                        new = next_points[status.ravel() == 1].reshape(-1, 2)
                        if len(old) >= 12 and correlation > 0.55:
                            matrix, _ = cv2.estimateAffinePartial2D(
                                old, new, method=cv2.RANSAC, ransacReprojThreshold=2.0
                            )
                            if matrix is not None:
                                dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
                                flows.append((timestamp, dx, dy, (dx * dx + dy * dy) ** 0.5))
            previous = (gray, histogram)
        cap.release()
        if not flows:
            return None

        directional = [item for item in flows if abs(item[1]) > 1.5]
        denominator = max(1, len(flows))
        up_direction = sum(max(0.0, item[2]) for item in directional) / denominator
        down_direction = sum(max(0.0, -item[2]) for item in directional) / denominator
        diagonal_fraction = sum(
            1 for _, dx, dy, _ in directional if abs(dx) > 1.5 and abs(dy) > 0.5
        ) / denominator
        final_window = [item for item in flows if anchor_seconds <= item[0] <= anchor_seconds + 3.0]
        static_fraction = sum(item[3] < 0.3 for item in final_window) / max(1, len(final_window))
        direction_balance = max(-1.0, min(1.0, (up_direction - down_direction) / 3.0))
        metrics = {
            "up_direction": direction_balance,
            "down_direction": -direction_balance,
            "diagonal_fraction": diagonal_fraction,
            "static_fraction": static_fraction,
            "transition_score": min(1.0, len(transitions) / 2.0),
        }
        score = self._score_metrics(metrics, constraints)
        return score, {
            "anchor_frame_index": frame_index,
            "anchor_seconds": round(anchor_seconds, 3),
            "transition_count": len(transitions),
            "transition_seconds": [round(item, 3) for item in transitions],
            "metrics": {key: round(value, 5) for key, value in metrics.items()},
        }
