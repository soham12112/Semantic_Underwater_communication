"""
Stage D: Candidate Subject Discovery + ROI Tracking

Discovers main subjects even when object detectors fail.
Uses saliency/structure analysis and optional tracking.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import ROIConfig
from schemas import ROIData, ROITrajectory


@dataclass
class ROICandidate:
    """A candidate region of interest."""
    x: int
    y: int
    width: int
    height: int
    score: float
    method: str  # How it was detected


class ROIDiscovery:
    """
    Stage D: Find and track main subjects using CPU-friendly methods.
    
    Methods:
    - Saliency/edge density for initial ROI discovery
    - CSRT or KLT tracking for temporal consistency
    """
    
    def __init__(self, config: Optional[ROIConfig] = None):
        self.config = config or ROIConfig()
        self._tracker: Optional[cv2.Tracker] = None
    
    def _compute_edge_density_map(
        self, 
        frame: np.ndarray,
        block_size: int = 32
    ) -> np.ndarray:
        """
        Compute edge density per block.
        
        Higher values indicate more detail/structure.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.config.edge_blur_size, self.config.edge_blur_size), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Compute block-wise density
        h, w = edges.shape
        blocks_y = h // block_size
        blocks_x = w // block_size
        
        density_map = np.zeros((blocks_y, blocks_x), dtype=np.float32)
        
        for by in range(blocks_y):
            for bx in range(blocks_x):
                y0 = by * block_size
                y1 = y0 + block_size
                x0 = bx * block_size
                x1 = x0 + block_size
                
                block = edges[y0:y1, x0:x1]
                density_map[by, bx] = np.sum(block > 0) / block.size
        
        return density_map
    
    def _apply_position_bias(
        self,
        density_map: np.ndarray
    ) -> np.ndarray:
        """
        Apply position-based bias (center and lower-half preference).
        
        ROVs typically frame subjects in center-lower area.
        """
        h, w = density_map.shape
        
        # Create center bias (Gaussian)
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        center_bias = np.exp(-(xx**2 + yy**2) / 0.5)
        center_bias = center_bias * self.config.center_bias_weight
        
        # Create lower-half bias
        lower_bias = np.zeros_like(density_map)
        lower_bias[h//2:, :] = self.config.lower_half_bias
        
        # Combine
        biased = density_map + center_bias + lower_bias
        
        return biased
    
    def _extract_top_rois(
        self,
        density_map: np.ndarray,
        frame_shape: Tuple[int, int],
        block_size: int = 32
    ) -> List[ROICandidate]:
        """
        Extract top ROI regions from density map.
        """
        h, w = frame_shape[:2]
        map_h, map_w = density_map.shape
        
        # Find local maxima
        dilated = cv2.dilate(density_map, np.ones((3, 3)))
        local_max = (density_map == dilated) & (density_map > 0.1)
        
        # Get coordinates of local maxima
        max_coords = np.where(local_max)
        scores = density_map[max_coords]
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        
        rois = []
        used_regions = set()
        
        for idx in sorted_idx[:self.config.max_roi_count * 3]:  # Extra candidates for filtering
            by = max_coords[0][idx]
            bx = max_coords[1][idx]
            score = scores[idx]
            
            # Skip if too close to existing ROI
            region_key = (by // 2, bx // 2)
            if region_key in used_regions:
                continue
            used_regions.add(region_key)
            
            # Convert block coords to pixel coords
            # Expand ROI to include neighboring blocks
            y0 = max(0, (by - 1) * block_size)
            y1 = min(h, (by + 2) * block_size)
            x0 = max(0, (bx - 1) * block_size)
            x1 = min(w, (bx + 2) * block_size)
            
            width = x1 - x0
            height = y1 - y0
            
            if width < self.config.min_roi_size or height < self.config.min_roi_size:
                continue
            
            rois.append(ROICandidate(
                x=x0,
                y=y0,
                width=width,
                height=height,
                score=float(score),
                method="edge_density"
            ))
            
            if len(rois) >= self.config.max_roi_count:
                break
        
        return rois
    
    def _contour_based_roi(
        self,
        frame: np.ndarray
    ) -> List[ROICandidate]:
        """
        Alternative ROI detection using contour analysis.
        
        Finds largest/most significant contours.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Threshold and find contours
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        rois = []
        h, w = frame.shape[:2]
        
        for contour in contours[:self.config.max_roi_count]:
            area = cv2.contourArea(contour)
            
            # Skip very small or very large contours
            if area < self.config.min_roi_size ** 2:
                continue
            if area > 0.8 * h * w:  # Skip if covering most of frame
                continue
            
            x, y, width, height = cv2.boundingRect(contour)
            
            # Compute score based on area and position
            center_x = x + width / 2
            center_y = y + height / 2
            center_dist = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
            max_dist = np.sqrt((w/2)**2 + (h/2)**2)
            center_score = 1 - (center_dist / max_dist)
            
            score = (area / (h * w)) * 0.5 + center_score * 0.5
            
            rois.append(ROICandidate(
                x=x,
                y=y,
                width=width,
                height=height,
                score=float(score),
                method="contour"
            ))
        
        return rois
    
    def discover_rois(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> List[ROIData]:
        """
        Discover ROIs in a single frame.
        
        Combines edge density and contour methods.
        """
        h, w = frame.shape[:2]
        
        # Method 1: Edge density
        density_map = self._compute_edge_density_map(frame)
        biased_map = self._apply_position_bias(density_map)
        density_rois = self._extract_top_rois(biased_map, (h, w))
        
        # Method 2: Contour-based
        contour_rois = self._contour_based_roi(frame)
        
        # Combine and deduplicate
        all_candidates = density_rois + contour_rois
        
        # Non-maximum suppression to remove overlapping ROIs
        final_rois = self._nms_rois(all_candidates)
        
        # Convert to ROIData
        results = []
        for roi in final_rois[:self.config.max_roi_count]:
            results.append(ROIData(
                timestamp=timestamp,
                x=roi.x,
                y=roi.y,
                width=roi.width,
                height=roi.height,
                confidence=roi.score
            ))
        
        return results
    
    def _nms_rois(
        self,
        candidates: List[ROICandidate],
        iou_threshold: float = 0.5
    ) -> List[ROICandidate]:
        """
        Non-maximum suppression to remove overlapping ROIs.
        """
        if not candidates:
            return []
        
        # Sort by score
        candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        
        keep = []
        
        for candidate in candidates:
            should_keep = True
            
            for kept in keep:
                iou = self._compute_iou(candidate, kept)
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(candidate)
        
        return keep
    
    def _compute_iou(
        self,
        roi1: ROICandidate,
        roi2: ROICandidate
    ) -> float:
        """Compute intersection over union between two ROIs."""
        x1 = max(roi1.x, roi2.x)
        y1 = max(roi1.y, roi2.y)
        x2 = min(roi1.x + roi1.width, roi2.x + roi2.width)
        y2 = min(roi1.y + roi1.height, roi2.y + roi2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = roi1.width * roi1.height
        area2 = roi2.width * roi2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_tracker(self, tracker_type: str):
        """
        Create a tracker, handling different OpenCV versions.
        
        OpenCV 4.5.1+ moved trackers to cv2.legacy module.
        """
        tracker_type = tracker_type.upper()
        
        # Try different API locations based on OpenCV version
        if tracker_type == "CSRT":
            # Try modern API first, then legacy
            if hasattr(cv2, 'TrackerCSRT_create'):
                return cv2.TrackerCSRT_create()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                return cv2.legacy.TrackerCSRT_create()
        elif tracker_type == "KCF":
            if hasattr(cv2, 'TrackerKCF_create'):
                return cv2.TrackerKCF_create()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
                return cv2.legacy.TrackerKCF_create()
        elif tracker_type == "MOSSE":
            if hasattr(cv2, 'TrackerMOSSE_create'):
                return cv2.TrackerMOSSE_create()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
                return cv2.legacy.TrackerMOSSE_create()
        
        # Fallback: try any available tracker
        for name in ['CSRT', 'KCF', 'MIL', 'MOSSE']:
            if hasattr(cv2, f'Tracker{name}_create'):
                return getattr(cv2, f'Tracker{name}_create')()
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, f'Tracker{name}_create'):
                return getattr(cv2.legacy, f'Tracker{name}_create')()
        
        # No tracker available
        return None
    
    def init_tracker(
        self,
        frame: np.ndarray,
        roi: ROIData
    ) -> bool:
        """
        Initialize tracker with a given ROI.
        
        Args:
            frame: Initial frame
            roi: ROI to track
            
        Returns: True if tracker initialized successfully
        """
        # Create tracker based on config
        self._tracker = self._create_tracker(self.config.tracker_type)
        
        if self._tracker is None:
            # No tracker available, skip tracking
            return False
        
        bbox = (roi.x, roi.y, roi.width, roi.height)
        try:
            return self._tracker.init(frame, bbox)
        except Exception:
            self._tracker = None
            return False
    
    def track(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Optional[ROIData]:
        """
        Track ROI in the next frame.
        
        Returns: Updated ROIData or None if tracking failed
        """
        if self._tracker is None:
            return None
        
        success, bbox = self._tracker.update(frame)
        
        if not success:
            return None
        
        x, y, w, h = [int(v) for v in bbox]
        
        return ROIData(
            timestamp=timestamp,
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=0.8  # Tracked ROI has moderate confidence
        )
    
    def track_sequence(
        self,
        frames: List[Tuple[float, np.ndarray]],
        initial_roi: Optional[ROIData] = None
    ) -> ROITrajectory:
        """
        Track ROI through a sequence of frames.
        
        Args:
            frames: List of (timestamp, frame) tuples
            initial_roi: Optional starting ROI; if None, will discover one
            
        Returns: ROITrajectory with tracked positions
        """
        if not frames:
            return ROITrajectory(rois=[])
        
        # Get initial ROI
        first_ts, first_frame = frames[0]
        
        if initial_roi is None:
            rois = self.discover_rois(first_frame, first_ts)
            if not rois:
                return ROITrajectory(rois=[])
            initial_roi = rois[0]  # Take best ROI
        
        # Initialize tracker
        if not self.init_tracker(first_frame, initial_roi):
            return ROITrajectory(rois=[initial_roi])
        
        tracked_rois = [initial_roi]
        
        # Track through remaining frames
        for timestamp, frame in frames[1:]:
            roi = self.track(frame, timestamp)
            if roi is not None:
                tracked_rois.append(roi)
            else:
                # Tracking lost, try to re-discover
                discovered = self.discover_rois(frame, timestamp)
                if discovered:
                    # Use closest discovered ROI
                    last_roi = tracked_rois[-1]
                    closest = min(
                        discovered,
                        key=lambda r: abs(r.x - last_roi.x) + abs(r.y - last_roi.y)
                    )
                    tracked_rois.append(closest)
                    # Re-initialize tracker
                    self.init_tracker(frame, closest)
                else:
                    break  # Lost track completely
        
        # Determine overall movement direction
        movement = self._analyze_trajectory(tracked_rois)
        
        return ROITrajectory(rois=tracked_rois, movement_direction=movement)
    
    def _analyze_trajectory(self, rois: List[ROIData]) -> Optional[str]:
        """Analyze trajectory to determine overall movement direction."""
        if len(rois) < 2:
            return None
        
        # Get start and end centers
        start_x = rois[0].x + rois[0].width // 2
        start_y = rois[0].y + rois[0].height // 2
        end_x = rois[-1].x + rois[-1].width // 2
        end_y = rois[-1].y + rois[-1].height // 2
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Size change (zoom proxy)
        start_area = rois[0].width * rois[0].height
        end_area = rois[-1].width * rois[-1].height
        size_change = end_area / start_area if start_area > 0 else 1.0
        
        directions = []
        
        if abs(dx) > 20:
            directions.append("right" if dx > 0 else "left")
        if abs(dy) > 20:
            directions.append("down" if dy > 0 else "up")
        if size_change > 1.2:
            directions.append("approaching")
        elif size_change < 0.8:
            directions.append("receding")
        
        return " + ".join(directions) if directions else "stationary"
    
    def crop_roi(
        self,
        frame: np.ndarray,
        roi: ROIData,
        padding: float = 0.1
    ) -> np.ndarray:
        """
        Crop ROI from frame with optional padding.
        
        Args:
            frame: Source frame
            roi: ROI to crop
            padding: Padding as fraction of ROI size
            
        Returns: Cropped image
        """
        h, w = frame.shape[:2]
        
        # Add padding
        pad_x = int(roi.width * padding)
        pad_y = int(roi.height * padding)
        
        x0 = max(0, roi.x - pad_x)
        y0 = max(0, roi.y - pad_y)
        x1 = min(w, roi.x + roi.width + pad_x)
        y1 = min(h, roi.y + roi.height + pad_y)
        
        return frame[y0:y1, x0:x1]


def discover_main_subject(
    frame: np.ndarray,
    config: Optional[ROIConfig] = None
) -> List[ROIData]:
    """
    Convenience function to discover main subject(s) in a frame.
    """
    discoverer = ROIDiscovery(config)
    return discoverer.discover_rois(frame)
