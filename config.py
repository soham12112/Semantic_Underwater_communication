"""
Configuration settings for the video-to-prompt pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class SamplingConfig:
    """Stage A: Video sampling parameters."""
    base_fps: float = 1.5  # Normal sampling rate
    burst_fps: float = 6.0  # High-rate sampling around events
    burst_window_sec: float = 0.5  # Window around detected events


@dataclass
class PreprocessConfig:
    """Stage B: Underwater preprocessing parameters."""
    max_side: int = 960  # Max dimension for enhanced frames
    motion_side: int = 320  # Max dimension for motion frames
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    apply_white_balance: bool = True
    apply_dehaze: bool = False  # Very mild dehaze
    dehaze_strength: float = 0.3


@dataclass
class MotionConfig:
    """Stage C: Motion estimation parameters."""
    orb_features: int = 500
    ransac_reproj_threshold: float = 5.0
    min_matches: int = 10
    # Motion thresholds for classification
    translation_threshold: float = 15.0  # pixels
    scale_threshold: float = 0.02  # percentage change
    static_threshold: float = 5.0  # below this = static


@dataclass
class ROIConfig:
    """Stage D: ROI discovery and tracking parameters."""
    method: str = "saliency"  # "saliency" or "tracker"
    # Saliency method params
    edge_blur_size: int = 5
    center_bias_weight: float = 0.3
    lower_half_bias: float = 0.2
    min_roi_size: int = 50
    max_roi_count: int = 3
    # Tracker params (if using CSRT)
    tracker_type: str = "CSRT"
    track_duration_sec: float = 3.0


@dataclass
class CaptionConfig:
    """Stage E: Captioning parameters."""
    model_name: str = "Salesforce/blip2-opt-2.7b"
    device: str = "cpu"  # or "cuda"
    max_new_tokens: int = 50
    frames_per_event: int = 5
    caption_roi: bool = True
    caption_global: bool = True


@dataclass
class LLMConfig:
    """Stage G: LLM prompt synthesis parameters."""
    # Provider options: "minimax", "openai", "anthropic"
    # MiniMax M2.1 recommended for best instruction following
    # Reference: https://www.minimax.io/news/minimax-m21
    provider: str = "minimax"
    # Model options:
    #   MiniMax: "MiniMax-M2.1", "MiniMax-M2.1-lightning", "MiniMax-M2"
    #   OpenAI: "gpt-4o", "gpt-4-turbo"
    #   Anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
    model: str = "MiniMax-M2.1"
    temperature: float = 0.3
    max_tokens: int = 1000


@dataclass 
class ContextTemplate:
    """Pre-provided context for LLM synthesis."""
    domain: str = "underwater"
    platform: str = "ROV"
    camera_style: str = "documentary GoPro wide-angle"
    priorities: List[str] = field(default_factory=lambda: [
        "realism",
        "smooth motion", 
        "stable color palette",
        "faithful reconstruction"
    ])
    forbidden: List[str] = field(default_factory=lambda: [
        "adding people unless detected",
        "adding fish/animals unless evidence",
        "unrealistic neon colors",
        "shaky/erratic motion"
    ])
    target_duration: str = "2-4 seconds"


@dataclass
class PipelineConfig:
    """Master configuration combining all stages."""
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextTemplate = field(default_factory=ContextTemplate)
    
    # Output settings
    output_dir: Path = Path("./output")
    save_intermediate: bool = True
    verbose: bool = True


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()
