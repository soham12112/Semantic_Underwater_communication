"""
Stage A: Video Ingestion + Sampling

Samples frames from video at configurable rates:
- Base rate (1-2 fps) for general scene understanding
- Burst rate (5-8 fps) around detected motion events
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass

from config import SamplingConfig
from schemas import SampledFrame


@dataclass
class VideoMetadata:
    """Basic video information."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


class VideoSampler:
    """
    Stage A: Sample frames from video at strategic intervals.
    
    Outputs sampled frames + timestamps for downstream processing.
    """
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
    
    def open(self, video_path: str) -> VideoMetadata:
        """Open video file and extract metadata."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self._metadata = VideoMetadata(
            path=str(path),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_sec=0.0
        )
        self._metadata.duration_sec = (
            self._metadata.frame_count / self._metadata.fps 
            if self._metadata.fps > 0 else 0.0
        )
        
        return self._metadata
    
    def close(self):
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    @property
    def metadata(self) -> Optional[VideoMetadata]:
        return self._metadata
    
    def _get_frame_at_index(self, frame_idx: int) -> Optional[np.ndarray]:
        """Seek to and read a specific frame."""
        if self._cap is None:
            return None
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        return frame if ret else None
    
    def _get_frame_at_time(self, timestamp_sec: float) -> Tuple[Optional[np.ndarray], int]:
        """Get frame at specific timestamp."""
        if self._cap is None or self._metadata is None:
            return None, -1
        
        frame_idx = int(timestamp_sec * self._metadata.fps)
        frame_idx = max(0, min(frame_idx, self._metadata.frame_count - 1))
        
        frame = self._get_frame_at_index(frame_idx)
        return frame, frame_idx
    
    def compute_base_sample_indices(self) -> List[Tuple[int, float]]:
        """
        Compute frame indices for base sampling rate.
        
        Returns: List of (frame_idx, timestamp) tuples
        """
        if self._metadata is None:
            return []
        
        interval_frames = int(self._metadata.fps / self.config.base_fps)
        interval_frames = max(1, interval_frames)
        
        indices = []
        for idx in range(0, self._metadata.frame_count, interval_frames):
            timestamp = idx / self._metadata.fps
            indices.append((idx, timestamp))
        
        return indices
    
    def compute_burst_indices(
        self, 
        event_times: List[float]
    ) -> List[Tuple[int, float]]:
        """
        Compute additional frame indices for burst sampling around events.
        
        Args:
            event_times: List of timestamps (seconds) where events occur
            
        Returns: List of (frame_idx, timestamp) tuples
        """
        if self._metadata is None:
            return []
        
        interval_frames = int(self._metadata.fps / self.config.burst_fps)
        interval_frames = max(1, interval_frames)
        
        window_frames = int(self.config.burst_window_sec * self._metadata.fps)
        
        indices_set = set()
        
        for event_t in event_times:
            event_frame = int(event_t * self._metadata.fps)
            start_frame = max(0, event_frame - window_frames)
            end_frame = min(self._metadata.frame_count, event_frame + window_frames)
            
            for idx in range(start_frame, end_frame, interval_frames):
                timestamp = idx / self._metadata.fps
                indices_set.add((idx, timestamp))
        
        return sorted(list(indices_set))
    
    def sample_frames(
        self,
        event_times: Optional[List[float]] = None,
        return_frames: bool = True
    ) -> Generator[Tuple[SampledFrame, Optional[np.ndarray]], None, None]:
        """
        Generator that yields sampled frames.
        
        Args:
            event_times: Optional list of event timestamps for burst sampling
            return_frames: If True, yield actual frame data; if False, just metadata
            
        Yields: (SampledFrame, frame_array) tuples
        """
        if self._metadata is None:
            return
        
        # Get base sampling indices
        base_indices = self.compute_base_sample_indices()
        base_set = set(idx for idx, _ in base_indices)
        
        # Get burst indices if events provided
        burst_indices = []
        if event_times:
            burst_indices = self.compute_burst_indices(event_times)
        
        # Combine and sort all indices
        all_indices = {}
        for idx, ts in base_indices:
            all_indices[idx] = (ts, False)  # (timestamp, is_burst)
        for idx, ts in burst_indices:
            if idx not in base_set:
                all_indices[idx] = (ts, True)
        
        sorted_indices = sorted(all_indices.items())
        
        # Yield frames
        for frame_idx, (timestamp, is_burst) in sorted_indices:
            frame = None
            if return_frames:
                frame = self._get_frame_at_index(frame_idx)
            
            sample = SampledFrame(
                timestamp=timestamp,
                frame_idx=frame_idx,
                is_burst=is_burst
            )
            
            yield sample, frame
    
    def sample_all(
        self,
        video_path: str,
        event_times: Optional[List[float]] = None
    ) -> Tuple[VideoMetadata, List[Tuple[SampledFrame, np.ndarray]]]:
        """
        Convenience method: open video and sample all frames at once.
        
        Returns: (metadata, list of (SampledFrame, frame_array) tuples)
        """
        metadata = self.open(video_path)
        
        samples = []
        for sample, frame in self.sample_frames(event_times, return_frames=True):
            if frame is not None:
                samples.append((sample, frame))
        
        self.close()
        return metadata, samples
    
    def get_keyframes(
        self,
        timestamps: List[float]
    ) -> List[Tuple[float, Optional[np.ndarray]]]:
        """
        Get specific frames at given timestamps.
        
        Args:
            timestamps: List of timestamps in seconds
            
        Returns: List of (timestamp, frame) tuples
        """
        results = []
        for ts in timestamps:
            frame, _ = self._get_frame_at_time(ts)
            results.append((ts, frame))
        return results


def sample_video(
    video_path: str,
    config: Optional[SamplingConfig] = None
) -> Tuple[VideoMetadata, List[Tuple[SampledFrame, np.ndarray]]]:
    """
    Convenience function to sample a video.
    
    Args:
        video_path: Path to video file
        config: Optional sampling configuration
        
    Returns: (metadata, samples) tuple
    """
    sampler = VideoSampler(config)
    return sampler.sample_all(video_path)
