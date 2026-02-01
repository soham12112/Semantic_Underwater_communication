"""
Stage E: Lightweight Semantics (Captioning)

Generates captions for:
- Global frames (environment/scene)
- ROI crops (main subject)

Supports:
- MiniMax Vision API (recommended for underwater - fast and accurate)
- BLIP-2 (slower, runs on CPU)
"""
import cv2
import numpy as np
import base64
import json
import os
import requests
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


class MiniMaxVisionCaptioner:
    """
    Use MiniMax Vision API to analyze frames and ROIs.
    
    Much faster than BLIP-2 on CPU and better for underwater scenes.
    Specifically trained to recognize statues, sculptures, marine life, etc.
    """
    
    def __init__(
        self, 
        model: str = "MiniMax-Text-01",
        scene_hint: str = ""
    ):
        self.api_key = os.getenv("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimax.io/v1/chat/completions"
        self.model = model
        self.scene_hint = scene_hint
        self._available = bool(self.api_key)
        
        if not self._available:
            logger.warning("MiniMax Vision not available: MINIMAX_API_KEY not set")
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 JPEG."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _call_api(self, user_content: str) -> Optional[str]:
        """Make MiniMax API call with vision content."""
        if not self._available:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing underwater video frames. Return ONLY valid JSON responses."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            result = response.json()
            
            if "error" in result:
                logger.error(f"MiniMax Vision API error: {result['error']}")
                return None
            
            if "choices" not in result or len(result["choices"]) == 0:
                logger.error(f"Unexpected MiniMax response: {result}")
                return None
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"MiniMax Vision API call failed: {e}")
            return None
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown."""
        text = text.strip()
        
        if text.startswith("```"):
            lines = text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = '\n'.join(lines)
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        return text
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        roi: Optional[ROIData] = None,
        timestamp: float = 0.0
    ) -> KeyframeCaption:
        """
        Analyze a frame using MiniMax Vision.
        
        Args:
            frame: Full frame image (BGR)
            roi: Optional ROI to focus on
            timestamp: Frame timestamp
            
        Returns: KeyframeCaption with descriptions
        """
        if not self._available:
            return KeyframeCaption(timestamp=timestamp)
        
        # Encode frame
        frame_b64 = self._encode_image(frame)
        
        # Build ROI info
        roi_info = ""
        if roi:
            roi_info = f"\nFocus on the region at bbox [{roi.x}, {roi.y}, {roi.x + roi.width}, {roi.y + roi.height}]."
        
        # Build prompt
        scene_context = f"\nScene context: {self.scene_hint}\n" if self.scene_hint else ""
        
        prompt = f"""Analyze this underwater video frame.
{scene_context}{roi_info}

Identify what you see and provide descriptions.

Labels to consider: ["fish", "shark", "ship", "human", "statue", "coral", "debris", "none"]
- "statue" = underwater sculpture, bust, figure, art installation (may have gestures)
- "human" = actual living person, diver  
- "debris" = wreckage, man-made objects on seabed

IMPORTANT: Look carefully for human-made objects like statues or sculptures.

Return ONLY valid JSON:
{{
    "global_description": "Brief description of the overall scene",
    "main_subject": {{
        "label": "statue|fish|coral|human|debris|none",
        "confidence": 0.0 to 1.0,
        "description": "Detailed description of the main subject"
    }},
    "environment": {{
        "water_color": "description of water color",
        "visibility": "clear|hazy|murky",
        "seafloor": "description of seafloor if visible"
    }}
}}"""

        user_content = f"[Image base64:{frame_b64}]\n\n{prompt}"
        
        response = self._call_api(user_content)
        
        if response is None:
            return KeyframeCaption(timestamp=timestamp)
        
        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            
            global_caption = data.get("global_description", "")
            
            # Build ROI caption from main subject
            roi_caption = None
            main_subject = data.get("main_subject", {})
            if main_subject:
                label = main_subject.get("label", "")
                desc = main_subject.get("description", "")
                conf = main_subject.get("confidence", 0)
                if label and label != "none":
                    roi_caption = f"{label}: {desc} (confidence: {conf:.0%})"
                elif desc:
                    roi_caption = desc
            
            # Extract detected objects
            detected = []
            if main_subject.get("label") and main_subject["label"] != "none":
                detected.append(main_subject["label"])
            
            return KeyframeCaption(
                timestamp=timestamp,
                global_caption=global_caption,
                roi_caption=roi_caption,
                detected_objects=detected
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MiniMax response: {e}")
            return KeyframeCaption(
                timestamp=timestamp,
                global_caption=response[:200] if response else None
            )
    
    def analyze_keyframes(
        self,
        frames: List[Tuple[float, np.ndarray]],
        rois: Optional[List[Optional[ROIData]]] = None
    ) -> List[KeyframeCaption]:
        """
        Analyze multiple keyframes.
        
        Args:
            frames: List of (timestamp, frame) tuples
            rois: Optional list of ROIs per frame
            
        Returns: List of KeyframeCaption
        """
        if not self._available:
            return [KeyframeCaption(timestamp=t) for t, _ in frames]
        
        if rois is None:
            rois = [None] * len(frames)
        
        results = []
        for i, (timestamp, frame) in enumerate(frames):
            roi = rois[i] if i < len(rois) else None
            caption = self.analyze_frame(frame, roi, timestamp)
            results.append(caption)
            logger.info(f"  Analyzed keyframe at {timestamp:.1f}s")
        
        return results


class SemanticCaptioner:
    """
    Stage E: Generate semantic captions for frames and ROIs.
    
    Supports:
    - MiniMax Vision API (fast, recommended for underwater)
    - BLIP-2 (slow CPU fallback)
    
    Provides weak signals about scene content that feed into
    the structured scene report.
    """
    
    def __init__(
        self, 
        config: Optional[CaptionConfig] = None,
        scene_hint: str = ""
    ):
        self.config = config or CaptionConfig()
        self._model_loaded = False
        self._minimax_captioner: Optional[MiniMaxVisionCaptioner] = None
        self.scene_hint = scene_hint
        
        # Initialize MiniMax Vision if configured
        if self.config.use_minimax_vision:
            self._minimax_captioner = MiniMaxVisionCaptioner(
                model=self.config.minimax_vision_model,
                scene_hint=scene_hint
            )
            if self._minimax_captioner.is_available:
                logger.info("Using MiniMax Vision for captioning (fast mode)")
            else:
                logger.info("MiniMax Vision not available, will use BLIP-2 fallback")
    
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
        
        Uses MiniMax Vision if available (fast), otherwise BLIP-2 (slow).
        
        Args:
            frames: List of (timestamp, frame) tuples
            rois: Optional list of ROIs (one per frame, or None)
            
        Returns: List of KeyframeCaption objects
        """
        # Try MiniMax Vision first (fast)
        if (self._minimax_captioner is not None and 
            self._minimax_captioner.is_available):
            logger.info("Using MiniMax Vision for keyframe analysis...")
            return self._minimax_captioner.analyze_keyframes(frames, rois)
        
        # Fallback to BLIP-2 (slow)
        logger.info("Using BLIP-2 for keyframe captioning (slow)...")
        
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
