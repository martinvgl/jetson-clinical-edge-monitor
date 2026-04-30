"""
Event detector for the Jetson Clinical Edge Monitor.

Takes 18-keypoint poses (COCO topology) and outputs high-level events:
  - FALL_DETECTED      : sudden drop of center-of-mass below threshold
  - PROLONGED_STILLNESS: no significant movement for N seconds
  - POSTURE_WARNING    : trunk angle deviates from vertical for N seconds

Pure geometry, no ML. Explainable, auditable, privacy-safe.
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# COCO 18-keypoint indices (trt_pose human_pose.json topology)
# ---------------------------------------------------------------------------
# 0 nose         5 left_shoulder    11 left_hip      15 left_eye
# 1 left_eye     6 right_shoulder   12 right_hip     16 right_ear
# 2 right_eye    7 left_elbow       13 left_knee     17 (neck if present)
# 3 left_ear     8 right_elbow      14 right_knee
# 4 right_ear    9 left_wrist
#                10 right_wrist

KP_NOSE = 0
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class EventType(Enum):
    NONE = "NONE"
    FALL_DETECTED = "FALL_DETECTED"
    PROLONGED_STILLNESS = "PROLONGED_STILLNESS"
    POSTURE_WARNING = "POSTURE_WARNING"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"


@dataclass
class Event:
    type: EventType
    severity: Severity
    timestamp: float  # unix epoch
    metadata: dict    # for debug / dev_log only, never published as-is


# ---------------------------------------------------------------------------
# Configuration (tune via configs/default.yaml later)
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    # Geometry
    fall_com_relative_height: float = 0.6   # CoM below 60% from top → suspicious
    fall_drop_speed: float = 0.25            # CoM y must drop > 25% in fall_window_sec
    fall_window_sec: float = 1.0             # over how long the drop must happen

    posture_angle_threshold_deg: float = 45  # trunk angle from vertical
    posture_min_duration_sec: float = 3.0    # must persist this long to fire

    stillness_window_sec: float = 30.0       # window over which we measure motion
    stillness_movement_threshold: float = 0.02  # normalized motion below this = still
    stillness_min_duration_sec: float = 30.0

    # Cooldowns to avoid event spam (seconds)
    fall_cooldown_sec: float = 10.0
    posture_cooldown_sec: float = 30.0
    stillness_cooldown_sec: float = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _midpoint(p1, p2):
    """Midpoint of two (x, y) tuples."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _trunk_angle_deg(shoulders_mid, hips_mid):
    """
    Angle of the trunk vector (hips → shoulders) relative to vertical.
    0° = perfectly upright, 90° = horizontal (lying down).
    Image y axis points down, so vertical-up is (0, -1).
    """
    import math
    dx = shoulders_mid[0] - hips_mid[0]
    dy = shoulders_mid[1] - hips_mid[1]
    # Angle vs vertical-up vector (0, -1)
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


def _extract_keypoints(pose, image_height):
    """
    Extract the keypoints we need from a pose.

    `pose` is expected as a dict {kp_index: (x, y)} where coords are in pixels,
    or None if a keypoint is missing.

    Returns a dict of derived quantities, normalized by image_height for
    resolution-invariance, or None if essential keypoints are missing.
    """
    required = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP]
    if not all(k in pose and pose[k] is not None for k in required):
        return None

    shoulders_mid = _midpoint(pose[KP_L_SHOULDER], pose[KP_R_SHOULDER])
    hips_mid = _midpoint(pose[KP_L_HIP], pose[KP_R_HIP])

    # Center of mass approximated as midpoint of shoulders+hips
    com = _midpoint(shoulders_mid, hips_mid)
    com_y_norm = com[1] / image_height  # 0 = top of image, 1 = bottom

    trunk_angle = _trunk_angle_deg(shoulders_mid, hips_mid)

    return {
        "com": com,
        "com_y_norm": com_y_norm,
        "trunk_angle_deg": trunk_angle,
        "shoulders_mid": shoulders_mid,
        "hips_mid": hips_mid,
    }


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Stateful detector. Feed it one (timestamp, pose, image_height) per frame
    via `update(...)` and it returns an Event or None.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()

        # Rolling histories: deques of (timestamp, value)
        # Sized generously; oldest entries pruned by time, not count.
        self._com_history: deque = deque(maxlen=600)        # (t, com_y_norm)
        self._motion_history: deque = deque(maxlen=600)     # (t, com_xy)
        self._posture_history: deque = deque(maxlen=300)    # (t, angle_deg)

        # Cooldown timestamps (next time we are allowed to re-fire)
        self._next_fall_ok: float = 0.0
        self._next_posture_ok: float = 0.0
        self._next_stillness_ok: float = 0.0

    def update(self, timestamp: float, pose: Optional[dict],
               image_height: int) -> Optional[Event]:
        """
        Process one frame.
        Returns an Event if one is fired this frame, else None.
        """
        if pose is None:
            return None

        features = _extract_keypoints(pose, image_height)
        if features is None:
            return None

        # Update histories
        self._com_history.append((timestamp, features["com_y_norm"]))
        self._motion_history.append((timestamp, features["com"]))
        self._posture_history.append((timestamp, features["trunk_angle_deg"]))

        # Prune old entries (keep the longest window we need)
        max_window = max(
            self.cfg.fall_window_sec,
            self.cfg.stillness_window_sec,
            self.cfg.posture_min_duration_sec,
        )
        cutoff = timestamp - max_window
        while self._com_history and self._com_history[0][0] < cutoff:
            self._com_history.popleft()
        while self._motion_history and self._motion_history[0][0] < cutoff:
            self._motion_history.popleft()
        while self._posture_history and self._posture_history[0][0] < cutoff:
            self._posture_history.popleft()

        # Run detectors in priority order: FALL > POSTURE > STILLNESS
        evt = self._detect_fall(timestamp, features)
        if evt:
            return evt

        evt = self._detect_posture(timestamp, features)
        if evt:
            return evt

        evt = self._detect_stillness(timestamp)
        if evt:
            return evt

        return None

    # -- detectors ----------------------------------------------------------

    def _detect_fall(self, t: float, features: dict) -> Optional[Event]:
        if t < self._next_fall_ok:
            return None

        # Need enough history covering the fall window
        window_start = t - self.cfg.fall_window_sec
        in_window = [(ts, y) for ts, y in self._com_history if ts >= window_start]
        if len(in_window) < 3:
            return None

        y_at_start = in_window[0][1]
        y_now = in_window[-1][1]
        drop = y_now - y_at_start  # positive if CoM moved down

        if (drop > self.cfg.fall_drop_speed and
                y_now > self.cfg.fall_com_relative_height):
            self._next_fall_ok = t + self.cfg.fall_cooldown_sec
            return Event(
                type=EventType.FALL_DETECTED,
                severity=Severity.ALERT,
                timestamp=t,
                metadata={
                    "com_y_drop": round(drop, 3),
                    "com_y_final": round(y_now, 3),
                    "window_sec": self.cfg.fall_window_sec,
                },
            )
        return None

    def _detect_posture(self, t: float, features: dict) -> Optional[Event]:
        if t < self._next_posture_ok:
            return None

        threshold = self.cfg.posture_angle_threshold_deg
        min_dur = self.cfg.posture_min_duration_sec

        # All recent angles must exceed threshold
        recent = [(ts, a) for ts, a in self._posture_history
                  if ts >= t - min_dur]
        if not recent:
            return None
        if recent[0][0] > t - min_dur + 0.1:
            # Not enough history yet
            return None
        if all(a > threshold for _, a in recent):
            self._next_posture_ok = t + self.cfg.posture_cooldown_sec
            avg_angle = sum(a for _, a in recent) / len(recent)
            return Event(
                type=EventType.POSTURE_WARNING,
                severity=Severity.WARNING,
                timestamp=t,
                metadata={
                    "avg_angle_deg": round(avg_angle, 1),
                    "duration_sec": round(t - recent[0][0], 1),
                },
            )
        return None

    def _detect_stillness(self, t: float) -> Optional[Event]:
        if t < self._next_stillness_ok:
            return None

        min_dur = self.cfg.stillness_min_duration_sec
        recent = [(ts, p) for ts, p in self._motion_history
                  if ts >= t - min_dur]
        if len(recent) < 10:
            return None
        if recent[0][0] > t - min_dur + 0.1:
            return None

        # Total path length traveled by CoM (in pixels), normalized
        total = 0.0
        for i in range(1, len(recent)):
            (_, p_prev) = recent[i - 1]
            (_, p_curr) = recent[i]
            dx = p_curr[0] - p_prev[0]
            dy = p_curr[1] - p_prev[1]
            total += (dx * dx + dy * dy) ** 0.5

        # Normalize by frame count to get "average pixel motion per frame"
        avg_motion = total / max(1, len(recent) - 1)
        # Rough normalization (assuming ~480 px frame height).
        # Tune via stillness_movement_threshold in config.
        avg_motion_norm = avg_motion / 480.0

        if avg_motion_norm < self.cfg.stillness_movement_threshold:
            self._next_stillness_ok = t + self.cfg.stillness_cooldown_sec
            return Event(
                type=EventType.PROLONGED_STILLNESS,
                severity=Severity.WARNING,
                timestamp=t,
                metadata={
                    "avg_motion_norm": round(avg_motion_norm, 4),
                    "duration_sec": round(t - recent[0][0], 1),
                },
            )
        return None