"""
Stage E: Lightweight Semantics (Captioning)

Generates captions for:
- Global frames (environment/scene)
- ROI crops (main subject)

Uses BLIP-2 or similar model as weak semantic signal.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import logging

from config import CaptionConfig
from schemas import KeyframeCaption, ROIData

logger = logging.getLogger(__name__)


# Lazy imports for optional dependencies
_TRANSFORMERS_AVAILABLE = False
_CAPTION_MODEL = None
_CAPTION_PROCESSOR = None


def _load_caption_model(config: CaptionConfig):
    """Lazy-load the captioning model."""
    global _TRANSFORMERS_AVAILABLE, _CAPTION_MODEL, _CAPTION_PROCESSOR
    
    if _CAPTION_MODEL is not None:
        return True
    
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        from PIL import Image
        
        _TRANSFORMERS_AVAILABLE = True
        
        logger.info(f"Loading caption model: {config.model_name}")
        
        _CAPTION_PROCESSOR = Blip2Processor.from_pretrained(config.model_name)
        _CAPTION_MODEL = Blip2ForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        
        if config.device == "cuda":
            _CAPTION_MODEL = _CAPTION_MODEL.cuda()
        
        logger.info("Caption model loaded successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"Captioning dependencies not available: {e}")
        logger.warning("Install with: pip install transformers torch")
        return False
    except Exception as e:
        logger.error(f"Failed to load caption model: {e}")
        return False


class SemanticCaptioner:
    """
    Stage E: Generate semantic captions for frames and ROIs.
    
    Provides weak signals about scene content that feed into
    the structured scene report.
    """
    
    def __init__(self, config: Optional[CaptionConfig] = None):
        self.config = config or CaptionConfig()
        self._model_loaded = False
    
    def _ensure_model(self) -> bool:
        """Ensure caption model is loaded."""
        if self._model_loaded:
            return True
        
        self._model_loaded = _load_caption_model(self.config)
        return self._model_loaded
    
    def _numpy_to_pil(self, image: np.ndarray):
        """Convert numpy array (BGR) to PIL Image (RGB)."""
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    
    def caption_image(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate caption for a single image.
        
        Args:
            image: BGR numpy array
            prompt: Optional prompt to guide captioning
            
        Returns: Caption string
        """
        if not self._ensure_model():
            return self._fallback_caption(image)
        
        try:
            import torch
            
            pil_image = self._numpy_to_pil(image)
            
            # Prepare inputs
            if prompt:
                inputs = _CAPTION_PROCESSOR(
                    pil_image, prompt, return_tensors="pt"
                )
            else:
                inputs = _CAPTION_PROCESSOR(
                    pil_image, return_tensors="pt"
                )
            
            if self.config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                outputs = _CAPTION_MODEL.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens
                )
            
            caption = _CAPTION_PROCESSOR.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            
            return caption
            
        except Exception as e:
            logger.error(f"Captioning failed: {e}")
            return self._fallback_caption(image)
    
    def _fallback_caption(self, image: np.ndarray) -> str:
        """
        Generate a basic fallback caption using image statistics.
        
        Used when ML model is not available.
        """
        h, w = image.shape[:2]
        
        # Basic color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        # Determine dominant color family
        if avg_sat < 30:
            color = "grayscale"
        elif avg_hue < 15 or avg_hue > 165:
            color = "reddish"
        elif avg_hue < 45:
            color = "yellowish"
        elif avg_hue < 75:
            color = "greenish"
        elif avg_hue < 105:
            color = "cyan/turquoise"
        elif avg_hue < 135:
            color = "bluish"
        else:
            color = "purple/magenta"
        
        # Brightness
        if avg_val < 60:
            brightness = "dark"
        elif avg_val < 150:
            brightness = "medium lit"
        else:
            brightness = "bright"
        
        # Edge density for complexity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.05:
            complexity = "detailed"
        elif edge_density > 0.02:
            complexity = "moderate detail"
        else:
            complexity = "low detail"
        
        return f"A {brightness}, {color} scene with {complexity}"
    
    def caption_frame_and_roi(
        self,
        frame: np.ndarray,
        roi: Optional[ROIData] = None,
        timestamp: float = 0.0
    ) -> KeyframeCaption:
        """
        Generate captions for both full frame and ROI.
        
        Args:
            frame: Full frame image
            roi: Optional ROI to crop and caption separately
            timestamp: Frame timestamp
            
        Returns: KeyframeCaption with global and ROI captions
        """
        global_caption = None
        roi_caption = None
        
        # Caption full frame
        if self.config.caption_global:
            global_caption = self.caption_image(
                frame,
                prompt="Describe this underwater scene:"
            )
        
        # Caption ROI if provided
        if self.config.caption_roi and roi is not None:
            # Crop ROI with padding
            h, w = frame.shape[:2]
            pad = 20
            x0 = max(0, roi.x - pad)
            y0 = max(0, roi.y - pad)
            x1 = min(w, roi.x + roi.width + pad)
            y1 = min(h, roi.y + roi.height + pad)
            
            roi_crop = frame[y0:y1, x0:x1]
            
            if roi_crop.size > 0:
                roi_caption = self.caption_image(
                    roi_crop,
                    prompt="Describe this object:"
                )
        
        return KeyframeCaption(
            timestamp=timestamp,
            global_caption=global_caption,
            roi_caption=roi_caption
        )
    
    def caption_keyframes(
        self,
        frames: List[Tuple[float, np.ndarray]],
        rois: Optional[List[Optional[ROIData]]] = None
    ) -> List[KeyframeCaption]:
        """
        Caption multiple keyframes.
        
        Args:
            frames: List of (timestamp, frame) tuples
            rois: Optional list of ROIs (one per frame, or None)
            
        Returns: List of KeyframeCaption objects
        """
        if rois is None:
            rois = [None] * len(frames)
        
        results = []
        
        for i, (timestamp, frame) in enumerate(frames):
            roi = rois[i] if i < len(rois) else None
            caption = self.caption_frame_and_roi(frame, roi, timestamp)
            results.append(caption)
        
        return results
    
    def extract_objects_from_caption(
        self,
        caption: str
    ) -> List[str]:
        """
        Extract mentioned objects from a caption.
        
        Simple keyword extraction (can be enhanced with NLP).
        """
        # Common underwater objects to look for
        underwater_objects = [
            "fish", "coral", "rock", "sand", "statue", "sculpture",
            "seaweed", "algae", "reef", "shipwreck", "anchor",
            "diver", "person", "animal", "creature", "shell",
            "crab", "octopus", "shark", "ray", "turtle", "jellyfish",
            "artifact", "ruin", "column", "bust", "figure"
        ]
        
        caption_lower = caption.lower()
        found = []
        
        for obj in underwater_objects:
            if obj in caption_lower:
                found.append(obj)
        
        return found
    
    def consolidate_captions(
        self,
        captions: List[KeyframeCaption]
    ) -> Dict[str, Any]:
        """
        Consolidate multiple captions into summary.
        
        Extracts common themes and objects mentioned.
        """
        global_texts = [c.global_caption for c in captions if c.global_caption]
        roi_texts = [c.roi_caption for c in captions if c.roi_caption]
        
        # Count object mentions
        all_objects = []
        for caption in global_texts + roi_texts:
            all_objects.extend(self.extract_objects_from_caption(caption))
        
        # Count frequencies
        object_counts = {}
        for obj in all_objects:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        # Sort by frequency
        sorted_objects = sorted(
            object_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "sample_global_captions": global_texts[:3],
            "sample_roi_captions": roi_texts[:3],
            "detected_objects": [obj for obj, _ in sorted_objects[:10]],
            "total_captions": len(captions)
        }


class YOLODetector:
    """
    Optional YOLO-based object detection.
    
    Used as supplementary signal when objects are detectable.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if YOLO is available."""
        try:
            # Try to import ultralytics YOLO
            from ultralytics import YOLO
            self._available = True
        except ImportError:
            logger.info("YOLO not available, skipping object detection")
            self._available = False
    
    def _load_model(self):
        """Lazy load YOLO model."""
        if self._model is not None or not self._available:
            return
        
        try:
            from ultralytics import YOLO
            
            if self.model_path:
                self._model = YOLO(self.model_path)
            else:
                # Use default YOLOv8 nano
                self._model = YOLO("yolov8n.pt")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            self._available = False
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Run object detection on image.
        
        Returns list of detections with class, confidence, bbox.
        """
        if not self._available:
            return []
        
        self._load_model()
        
        if self._model is None:
            return []
        
        try:
            results = self._model(image, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    cls_name = result.names[cls_id]
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def check_for_people(self, detections: List[Dict]) -> bool:
        """Check if any people detected."""
        person_classes = {"person", "man", "woman", "child", "diver"}
        for det in detections:
            if det["class"].lower() in person_classes:
                return True
        return False
    
    def check_for_fish(self, detections: List[Dict]) -> Optional[bool]:
        """Check if fish detected (returns None if YOLO not available)."""
        if not self._available:
            return None
        
        fish_classes = {"fish", "shark", "ray", "dolphin", "whale"}
        for det in detections:
            if det["class"].lower() in fish_classes:
                return True
        return False


def caption_frames(
    frames: List[Tuple[float, np.ndarray]],
    config: Optional[CaptionConfig] = None
) -> List[KeyframeCaption]:
    """
    Convenience function to caption multiple frames.
    """
    captioner = SemanticCaptioner(config)
    return captioner.caption_keyframes(frames)
