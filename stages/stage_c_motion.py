"""
Stage C: Motion + Shot-Type Inference

Analyzes camera motion between frames to determine:
- Push in / Pull out (dolly/zoom)
- Truck left / right (lateral movement)
- Pedestal up / down (vertical movement)
- Static holds
- Pan and tilt (rotation)
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from config import MotionConfig
from schemas import MotionSegment


class MotionType(Enum):
    """Types of camera motion."""
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    TRUCK_LEFT = "truck_left"
    TRUCK_RIGHT = "truck_right"
    PEDESTAL_UP = "pedestal_up"
    PEDESTAL_DOWN = "pedestal_down"
    STATIC_HOLD = "static_hold"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"


@dataclass
class FrameMotion:
    """Motion data between two consecutive frames."""
    timestamp: float
    translation_x: float  # Positive = moving right
    translation_y: float  # Positive = moving down
    scale: float          # >1 = zoom in, <1 = zoom out
    rotation: float       # Degrees, positive = clockwise
    magnitude: float      # Overall motion magnitude
    labels: List[str]     # Motion type labels


class MotionAnalyzer:
    """
    Stage C: Analyze camera motion between frames.
    
    Uses ORB features + RANSAC homography to estimate
    global camera motion, then classifies into shot types.
    """
    
    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()
        
        # Initialize ORB detector
        self._orb = cv2.ORB_create(nfeatures=self.config.orb_features)
        
        # Initialize matcher
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def _extract_features(
        self, 
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ORB keypoints and descriptors."""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        
        if keypoints is None or len(keypoints) == 0:
            return np.array([]), None
        
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return pts, descriptors
    
    def _match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        """Match features between two frames."""
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        matches = self._bf.match(desc1, desc2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        return matches[:min(len(matches), self.config.orb_features // 2)]
    
    def _estimate_homography(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        matches: List[cv2.DMatch]
    ) -> Optional[np.ndarray]:
        """Estimate homography between matched points using RANSAC."""
        if len(matches) < self.config.min_matches:
            return None
        
        src_pts = np.float32([pts1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([pts2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.config.ransac_reproj_threshold
        )
        
        return H
    
    def _decompose_homography(
        self, 
        H: np.ndarray,
        frame_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """
        Decompose homography into translation, scale, and rotation.
        
        Returns: (tx, ty, scale, rotation_degrees)
        """
        if H is None:
            return 0.0, 0.0, 1.0, 0.0
        
        h, w = frame_size
        
        # Extract translation (from last column)
        tx = H[0, 2]
        ty = H[1, 2]
        
        # Extract scale (from diagonal elements, approximate)
        sx = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        sy = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
        scale = (sx + sy) / 2.0
        
        # Extract rotation (from upper-left 2x2)
        rotation_rad = np.arctan2(H[1, 0], H[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        
        return tx, ty, scale, rotation_deg
    
    def _classify_motion(
        self,
        tx: float,
        ty: float,
        scale: float,
        rotation: float
    ) -> List[str]:
        """
        Classify motion parameters into semantic labels.
        
        Returns list of applicable motion labels.
        """
        labels = []
        
        # Check scale (push/pull)
        scale_change = abs(scale - 1.0)
        if scale_change > self.config.scale_threshold:
            if scale > 1.0:
                labels.append(MotionType.PUSH_IN.value)
            else:
                labels.append(MotionType.PULL_OUT.value)
        
        # Check horizontal translation (truck)
        if abs(tx) > self.config.translation_threshold:
            if tx > 0:
                labels.append(MotionType.TRUCK_LEFT.value)
            else:
                labels.append(MotionType.TRUCK_RIGHT.value)
        
        # Check vertical translation (pedestal)
        if abs(ty) > self.config.translation_threshold:
            if ty > 0:
                labels.append(MotionType.PEDESTAL_UP.value)
            else:
                labels.append(MotionType.PEDESTAL_DOWN.value)
        
        # Check rotation (pan)
        if abs(rotation) > 2.0:  # degrees threshold
            if rotation > 0:
                labels.append(MotionType.PAN_RIGHT.value)
            else:
                labels.append(MotionType.PAN_LEFT.value)
        
        # If no significant motion, it's static
        if not labels:
            total_motion = np.sqrt(tx**2 + ty**2) + scale_change * 100
            if total_motion < self.config.static_threshold:
                labels.append(MotionType.STATIC_HOLD.value)
        
        return labels if labels else [MotionType.STATIC_HOLD.value]
    
    def analyze_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestamp: float
    ) -> FrameMotion:
        """
        Analyze motion between two consecutive frames.
        
        Args:
            frame1: First frame (earlier)
            frame2: Second frame (later)
            timestamp: Timestamp of frame2
            
        Returns: FrameMotion with motion parameters and labels
        """
        # Extract features
        pts1, desc1 = self._extract_features(frame1)
        pts2, desc2 = self._extract_features(frame2)
        
        # Match features
        matches = self._match_features(desc1, desc2)
        
        # Estimate homography
        H = self._estimate_homography(pts1, pts2, matches)
        
        # Get frame size
        h, w = frame1.shape[:2]
        
        # Decompose homography
        tx, ty, scale, rotation = self._decompose_homography(H, (h, w))
        
        # Classify motion
        labels = self._classify_motion(tx, ty, scale, rotation)
        
        # Compute overall magnitude
        magnitude = np.sqrt(tx**2 + ty**2) + abs(scale - 1.0) * 100
        
        return FrameMotion(
            timestamp=timestamp,
            translation_x=tx,
            translation_y=ty,
            scale=scale,
            rotation=rotation,
            magnitude=magnitude,
            labels=labels
        )
    
    def analyze_sequence(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[FrameMotion]:
        """
        Analyze motion across a sequence of frames.
        
        Args:
            frames: List of (timestamp, frame_array) tuples, sorted by time
            
        Returns: List of FrameMotion (one per consecutive pair)
        """
        if len(frames) < 2:
            return []
        
        motions = []
        for i in range(1, len(frames)):
            t_prev, frame_prev = frames[i - 1]
            t_curr, frame_curr = frames[i]
            
            motion = self.analyze_pair(frame_prev, frame_curr, t_curr)
            motions.append(motion)
        
        return motions
    
    def segment_motion(
        self,
        motions: List[FrameMotion],
        min_segment_duration: float = 0.3
    ) -> List[MotionSegment]:
        """
        Group consecutive frames with similar motion into segments.
        
        Args:
            motions: List of FrameMotion from analyze_sequence
            min_segment_duration: Minimum segment duration in seconds
            
        Returns: List of MotionSegment
        """
        if not motions:
            return []
        
        segments = []
        current_labels = set(motions[0].labels)
        segment_start = motions[0].timestamp - 0.1  # Approximate start
        magnitudes = [motions[0].magnitude]
        
        for i in range(1, len(motions)):
            motion = motions[i]
            motion_labels = set(motion.labels)
            
            # Check if motion type changed significantly
            if motion_labels != current_labels:
                # Save current segment
                segment_end = motion.timestamp
                
                if segment_end - segment_start >= min_segment_duration:
                    segments.append(MotionSegment(
                        t0=segment_start,
                        t1=segment_end,
                        labels=list(current_labels),
                        magnitude=np.mean(magnitudes),
                        smoothness=self._compute_smoothness(magnitudes)
                    ))
                
                # Start new segment
                current_labels = motion_labels
                segment_start = motion.timestamp
                magnitudes = [motion.magnitude]
            else:
                magnitudes.append(motion.magnitude)
        
        # Don't forget the last segment
        if magnitudes:
            segments.append(MotionSegment(
                t0=segment_start,
                t1=motions[-1].timestamp,
                labels=list(current_labels),
                magnitude=np.mean(magnitudes),
                smoothness=self._compute_smoothness(magnitudes)
            ))
        
        return segments
    
    def _compute_smoothness(self, magnitudes: List[float]) -> float:
        """
        Compute smoothness score from motion magnitudes.
        
        Returns value between 0 (jerky) and 1 (smooth).
        """
        if len(magnitudes) < 2:
            return 1.0
        
        # Compute variance of differences
        diffs = np.diff(magnitudes)
        variance = np.var(diffs)
        
        # Normalize to 0-1 range (inverse of variance)
        smoothness = 1.0 / (1.0 + variance * 0.01)
        
        return float(smoothness)


def analyze_video_motion(
    frames: List[Tuple[float, np.ndarray]],
    config: Optional[MotionConfig] = None
) -> List[MotionSegment]:
    """
    Convenience function to analyze motion in a video.
    
    Args:
        frames: List of (timestamp, frame_array) tuples
        config: Optional motion configuration
        
    Returns: List of motion segments
    """
    analyzer = MotionAnalyzer(config)
    motions = analyzer.analyze_sequence(frames)
    return analyzer.segment_motion(motions)
