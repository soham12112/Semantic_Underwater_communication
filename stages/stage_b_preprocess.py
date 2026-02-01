"""
Stage B: Underwater Preprocessing

Applies image enhancement for underwater footage:
- Gray-world white balance
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Optional mild dehazing
- Downscaling for consistency and speed
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

from config import PreprocessConfig


@dataclass
class EnhancedFrameResult:
    """Result from preprocessing a single frame."""
    timestamp: float
    frame_idx: int
    enhanced: np.ndarray  # Full resolution enhanced frame
    lowres: np.ndarray    # Low resolution for motion analysis


class UnderwaterPreprocessor:
    """
    Stage B: Enhance underwater footage for better analysis.
    
    Handles color correction, contrast enhancement, and creates
    multiple resolution outputs for different pipeline stages.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size)
        )
    
    def gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gray-world white balance assumption.
        
        Assumes the average color in the scene should be gray,
        then adjusts RGB channels to achieve this.
        """
        # Convert to float for precision
        img_float = image.astype(np.float32)
        
        # Calculate mean for each channel
        b_mean = np.mean(img_float[:, :, 0])
        g_mean = np.mean(img_float[:, :, 1])
        r_mean = np.mean(img_float[:, :, 2])
        
        # Overall mean (gray target)
        gray_mean = (b_mean + g_mean + r_mean) / 3.0
        
        # Avoid division by zero
        eps = 1e-6
        
        # Scale factors
        b_scale = gray_mean / (b_mean + eps)
        g_scale = gray_mean / (g_mean + eps)
        r_scale = gray_mean / (r_mean + eps)
        
        # Apply scaling
        result = img_float.copy()
        result[:, :, 0] = np.clip(result[:, :, 0] * b_scale, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * g_scale, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * r_scale, 0, 255)
        
        return result.astype(np.uint8)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to improve local contrast.
        
        Works in LAB color space to preserve color while
        enhancing luminance contrast.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def mild_dehaze(self, image: np.ndarray) -> np.ndarray:
        """
        Apply very mild dehazing effect.
        
        Uses dark channel prior concept but with reduced strength
        to avoid over-processing underwater scenes.
        """
        strength = self.config.dehaze_strength
        
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Simple atmospheric light estimation (brightest region)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        if np.sum(bright_mask) > 0:
            atm_light = np.mean(img_float[bright_mask > 0], axis=0)
        else:
            atm_light = np.array([0.9, 0.9, 0.9])
        
        # Dark channel (simplified)
        dark = np.min(img_float, axis=2)
        dark = cv2.erode(dark, np.ones((5, 5)))
        
        # Transmission estimate
        transmission = 1.0 - strength * dark
        transmission = np.clip(transmission, 0.1, 1.0)
        
        # Dehaze
        result = np.zeros_like(img_float)
        for c in range(3):
            result[:, :, c] = (
                (img_float[:, :, c] - atm_light[c]) / 
                np.maximum(transmission, 0.1) + atm_light[c]
            )
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result
    
    def resize_max_side(
        self, 
        image: np.ndarray, 
        max_side: int
    ) -> np.ndarray:
        """Resize image so largest side equals max_side."""
        h, w = image.shape[:2]
        
        if max(h, w) <= max_side:
            return image
        
        if w > h:
            new_w = max_side
            new_h = int(h * max_side / w)
        else:
            new_h = max_side
            new_w = int(w * max_side / h)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def process_frame(
        self, 
        frame: np.ndarray,
        timestamp: float = 0.0,
        frame_idx: int = 0
    ) -> EnhancedFrameResult:
        """
        Apply full preprocessing pipeline to a single frame.
        
        Args:
            frame: Input BGR image
            timestamp: Frame timestamp in seconds
            frame_idx: Original frame index
            
        Returns: EnhancedFrameResult with enhanced and lowres versions
        """
        # Step 1: White balance
        if self.config.apply_white_balance:
            balanced = self.gray_world_white_balance(frame)
        else:
            balanced = frame
        
        # Step 2: CLAHE
        enhanced = self.apply_clahe(balanced)
        
        # Step 3: Optional dehaze
        if self.config.apply_dehaze:
            enhanced = self.mild_dehaze(enhanced)
        
        # Step 4: Resize to max side
        enhanced_resized = self.resize_max_side(enhanced, self.config.max_side)
        
        # Step 5: Create low-res version for motion analysis
        lowres = self.resize_max_side(enhanced, self.config.motion_side)
        
        return EnhancedFrameResult(
            timestamp=timestamp,
            frame_idx=frame_idx,
            enhanced=enhanced_resized,
            lowres=lowres
        )
    
    def process_batch(
        self,
        frames: List[Tuple[float, int, np.ndarray]]
    ) -> List[EnhancedFrameResult]:
        """
        Process multiple frames.
        
        Args:
            frames: List of (timestamp, frame_idx, frame_array) tuples
            
        Returns: List of EnhancedFrameResult
        """
        results = []
        for timestamp, frame_idx, frame in frames:
            result = self.process_frame(frame, timestamp, frame_idx)
            results.append(result)
        return results
    
    def estimate_water_color(self, frame: np.ndarray) -> str:
        """
        Estimate dominant water color from frame.
        
        Returns a descriptive string like "turquoise-green" or "deep blue".
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get average hue (avoiding very dark/bright regions)
        mask = (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 230)
        
        if np.sum(mask) < 100:
            return "unknown"
        
        avg_hue = np.mean(hsv[:, :, 0][mask])
        avg_sat = np.mean(hsv[:, :, 1][mask])
        
        # Map hue to color description
        if avg_sat < 30:
            return "gray-murky"
        elif avg_hue < 15 or avg_hue > 165:
            return "reddish-brown (sediment)"
        elif avg_hue < 35:
            return "sandy-yellow"
        elif avg_hue < 75:
            return "green-murky"
        elif avg_hue < 95:
            return "turquoise-green"
        elif avg_hue < 115:
            return "cyan-clear"
        elif avg_hue < 135:
            return "deep blue"
        else:
            return "blue-purple"
    
    def estimate_visibility(self, frame: np.ndarray) -> str:
        """
        Estimate visibility conditions from frame contrast.
        
        Returns: "clear", "slight haze", "moderate haze", or "poor visibility"
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute contrast metrics
        std_dev = np.std(gray)
        
        # Edge density as visibility proxy
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combined assessment
        if std_dev > 50 and edge_density > 0.05:
            return "clear"
        elif std_dev > 35 and edge_density > 0.03:
            return "slight haze"
        elif std_dev > 20:
            return "moderate haze"
        else:
            return "poor visibility"
    
    def detect_particulate(self, frame: np.ndarray) -> str:
        """
        Detect presence of particulate matter (marine snow, sediment).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for small bright spots
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Count small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
        
        # Count small bright spots (particles)
        particle_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 2 < area < 50:  # Small spots
                particle_count += 1
        
        # Normalize by image size
        particle_density = particle_count / (gray.size / 10000)
        
        if particle_density > 5:
            return "heavy particulate"
        elif particle_density > 2:
            return "visible particulate"
        elif particle_density > 0.5:
            return "light particulate"
        else:
            return "minimal particulate"


def preprocess_frame(
    frame: np.ndarray,
    config: Optional[PreprocessConfig] = None
) -> EnhancedFrameResult:
    """Convenience function to preprocess a single frame."""
    preprocessor = UnderwaterPreprocessor(config)
    return preprocessor.process_frame(frame)
