"""
JSON schemas for scene reports and LLM outputs.
Uses Pydantic for validation and serialization.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# Motion Labels
# ============================================================================

class MotionLabel(str, Enum):
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    TRUCK_LEFT = "truck_left"
    TRUCK_RIGHT = "truck_right"
    PEDESTAL_UP = "pedestal_up"
    PEDESTAL_DOWN = "pedestal_down"
    STATIC_HOLD = "static_hold"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"


# ============================================================================
# Stage Outputs
# ============================================================================

class SampledFrame(BaseModel):
    """Output from Stage A: A single sampled frame."""
    timestamp: float = Field(..., description="Time in seconds")
    frame_idx: int = Field(..., description="Original frame index in video")
    is_burst: bool = Field(False, description="Whether from burst sampling")


class EnhancedFrame(BaseModel):
    """Output from Stage B: Enhanced frame data."""
    timestamp: float
    frame_idx: int
    enhanced_path: Optional[str] = Field(None, description="Path to enhanced frame")
    lowres_path: Optional[str] = Field(None, description="Path to low-res motion frame")


class MotionSegment(BaseModel):
    """Output from Stage C: A segment of consistent camera motion."""
    t0: float = Field(..., description="Start time in seconds")
    t1: float = Field(..., description="End time in seconds")
    labels: List[str] = Field(..., description="Motion labels for this segment")
    magnitude: float = Field(0.0, description="Average motion magnitude")
    smoothness: float = Field(1.0, description="Motion smoothness score 0-1")
    
    @property
    def duration(self) -> float:
        return self.t1 - self.t0
    
    @property
    def label_str(self) -> str:
        return " + ".join(self.labels) if self.labels else "static_hold"


class ROIData(BaseModel):
    """Output from Stage D: Region of interest data."""
    timestamp: float
    x: int = Field(..., description="Top-left x coordinate")
    y: int = Field(..., description="Top-left y coordinate")
    width: int
    height: int
    confidence: float = Field(0.0, description="Detection confidence")
    crop_path: Optional[str] = Field(None, description="Path to cropped ROI image")


class ROITrajectory(BaseModel):
    """Trajectory of tracked ROI over time."""
    rois: List[ROIData]
    movement_direction: Optional[str] = Field(None, description="Overall movement")
    
    @property
    def center_positions(self) -> List[tuple]:
        return [(r.x + r.width//2, r.y + r.height//2) for r in self.rois]


class KeyframeCaption(BaseModel):
    """Output from Stage E: Captions for a keyframe."""
    timestamp: float
    global_caption: Optional[str] = Field(None, description="Caption of full frame")
    roi_caption: Optional[str] = Field(None, description="Caption of ROI crop")
    detected_objects: List[str] = Field(default_factory=list)


# ============================================================================
# Scene Report (Stage F Output)
# ============================================================================

class ContextInfo(BaseModel):
    """Contextual information about the video."""
    domain: str = "underwater"
    platform: str = "ROV"
    camera_style: str = "documentary GoPro"
    constraints: List[str] = Field(default_factory=lambda: ["realistic colors", "smooth motion"])


class EnvironmentInfo(BaseModel):
    """Environment description from visual analysis."""
    water_color: str = Field("unknown", description="Estimated water color")
    visibility: str = Field("unknown", description="Visibility conditions")
    particulate: str = Field("unknown", description="Particulate matter presence")
    seafloor: List[str] = Field(default_factory=list, description="Seafloor features")
    lighting: str = Field("ambient", description="Lighting conditions")


class MainSubject(BaseModel):
    """Main subject hypothesis from ROI analysis."""
    hypothesis: str = Field("unknown", description="Best guess at main subject")
    appearance: List[str] = Field(default_factory=list, description="Appearance descriptors")
    notable_details: List[str] = Field(default_factory=list)
    confidence: float = Field(0.0, description="Confidence in hypothesis 0-1")


class NegativeEvidence(BaseModel):
    """Things explicitly NOT detected (useful for negative prompts)."""
    people_detected: bool = False
    fish_detected: Optional[bool] = None  # None = unknown
    text_detected: bool = False
    artificial_light: bool = False


class SceneReport(BaseModel):
    """
    Complete scene report (Stage F output).
    This is the structured input to the LLM.
    """
    video_path: str
    duration_sec: float
    
    context: ContextInfo = Field(default_factory=ContextInfo)
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    main_subject: MainSubject = Field(default_factory=MainSubject)
    
    camera_motion: List[MotionSegment] = Field(default_factory=list)
    negatives: NegativeEvidence = Field(default_factory=NegativeEvidence)
    keyframes: List[KeyframeCaption] = Field(default_factory=list)
    
    # Metadata
    processing_notes: List[str] = Field(default_factory=list)
    
    def to_llm_payload(self) -> dict:
        """Convert to dict suitable for LLM input."""
        return self.model_dump(exclude_none=True)


# ============================================================================
# LLM Output (Stage G Output)
# ============================================================================

class LLMPromptOutput(BaseModel):
    """
    Output from LLM prompt synthesis (Stage G).
    """
    final_prompt: str = Field(..., description="Main generation prompt")
    negative_prompt: str = Field("", description="Negative prompt for generation")
    confidence_notes: List[str] = Field(
        default_factory=list, 
        description="Notes about confidence/uncertainty"
    )
    used_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence from scene report that was used"
    )
    suggested_duration_sec: float = Field(3.0, description="Suggested clip duration")
    style_tags: List[str] = Field(default_factory=list, description="Style descriptors")


# ============================================================================
# Full Pipeline Output
# ============================================================================

class PipelineOutput(BaseModel):
    """Complete output from the video-to-prompt pipeline."""
    input_video: str
    scene_report: SceneReport
    llm_output: LLMPromptOutput
    
    # Intermediate artifacts (optional)
    sampled_frame_count: int = 0
    motion_segment_count: int = 0
    keyframe_count: int = 0
    
    processing_time_sec: float = 0.0
