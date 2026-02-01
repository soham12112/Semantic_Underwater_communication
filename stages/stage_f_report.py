"""
Stage F: Rule-Based Scene Report Assembler

Converts noisy outputs from previous stages into a structured,
consistent JSON report suitable for LLM consumption.

This is a CRITICAL step: the quality of the LLM output depends
heavily on how well we structure the evidence.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from config import PipelineConfig, ContextTemplate
from schemas import (
    SceneReport, ContextInfo, EnvironmentInfo, MainSubject,
    NegativeEvidence, MotionSegment, KeyframeCaption, ROIData
)

logger = logging.getLogger(__name__)


@dataclass
class StageOutputs:
    """Container for all outputs from previous stages."""
    video_path: str
    duration_sec: float
    
    # From Stage B
    water_color: Optional[str] = None
    visibility: Optional[str] = None
    particulate: Optional[str] = None
    
    # From Stage C
    motion_segments: List[MotionSegment] = None
    
    # From Stage D
    main_rois: List[ROIData] = None
    roi_movement: Optional[str] = None
    
    # From Stage E
    keyframe_captions: List[KeyframeCaption] = None
    detected_objects: List[str] = None
    people_detected: bool = False
    fish_detected: Optional[bool] = None
    
    def __post_init__(self):
        if self.motion_segments is None:
            self.motion_segments = []
        if self.main_rois is None:
            self.main_rois = []
        if self.keyframe_captions is None:
            self.keyframe_captions = []
        if self.detected_objects is None:
            self.detected_objects = []


class SceneReportAssembler:
    """
    Stage F: Assemble structured scene report from stage outputs.
    
    Key responsibilities:
    - Consolidate and deduplicate information
    - Resolve conflicts between different signals
    - Generate consistent descriptions
    - Prepare evidence for LLM consumption
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
    
    def assemble(self, outputs: StageOutputs) -> SceneReport:
        """
        Assemble complete scene report from stage outputs.
        
        Args:
            outputs: StageOutputs container with all stage results
            
        Returns: Structured SceneReport
        """
        report = SceneReport(
            video_path=outputs.video_path,
            duration_sec=outputs.duration_sec
        )
        
        # Build each section
        report.context = self._build_context()
        report.environment = self._build_environment(outputs)
        report.main_subject = self._build_main_subject(outputs)
        report.camera_motion = self._process_motion_segments(outputs.motion_segments)
        report.negatives = self._build_negatives(outputs)
        report.keyframes = self._select_best_keyframes(outputs.keyframe_captions)
        report.processing_notes = self._generate_notes(outputs)
        
        return report
    
    def _build_context(self) -> ContextInfo:
        """Build context section from configuration."""
        ctx = self.config.context
        return ContextInfo(
            domain=ctx.domain,
            platform=ctx.platform,
            camera_style=ctx.camera_style,
            constraints=ctx.priorities[:4]  # Top 4 priorities
        )
    
    def _build_environment(self, outputs: StageOutputs) -> EnvironmentInfo:
        """Build environment description from preprocessing results."""
        env = EnvironmentInfo()
        
        # Water color
        if outputs.water_color:
            env.water_color = outputs.water_color
        else:
            env.water_color = "unknown (analysis pending)"
        
        # Visibility
        if outputs.visibility:
            env.visibility = outputs.visibility
        
        # Particulate
        if outputs.particulate:
            env.particulate = outputs.particulate
        
        # Extract seafloor features from captions
        env.seafloor = self._extract_seafloor_features(outputs.keyframe_captions)
        
        # Lighting (infer from captions or default)
        env.lighting = self._infer_lighting(outputs.keyframe_captions)
        
        return env
    
    def _extract_seafloor_features(
        self,
        captions: List[KeyframeCaption]
    ) -> List[str]:
        """Extract seafloor features mentioned in captions."""
        seafloor_keywords = {
            "sand": "sandy bottom",
            "rock": "rocky substrate",
            "coral": "coral formations",
            "rubble": "coral rubble",
            "mud": "muddy bottom",
            "gravel": "gravel bed",
            "reef": "reef structure",
            "seagrass": "seagrass bed",
            "sediment": "soft sediment"
        }
        
        found = set()
        
        for caption in captions:
            text = ""
            if caption.global_caption:
                text += caption.global_caption.lower()
            
            for keyword, description in seafloor_keywords.items():
                if keyword in text:
                    found.add(description)
        
        return list(found)[:5]  # Limit to top 5
    
    def _infer_lighting(self, captions: List[KeyframeCaption]) -> str:
        """Infer lighting conditions from captions."""
        lighting_keywords = {
            "bright": "well-lit",
            "dark": "low ambient light",
            "spotlight": "artificial spotlight",
            "sunlight": "natural sunlight",
            "dim": "dim ambient",
            "shadowy": "shadowy conditions",
            "clear": "clear natural light"
        }
        
        for caption in captions:
            if caption.global_caption:
                text = caption.global_caption.lower()
                for keyword, description in lighting_keywords.items():
                    if keyword in text:
                        return description
        
        return "ambient underwater lighting"
    
    def _build_main_subject(self, outputs: StageOutputs) -> MainSubject:
        """Build main subject hypothesis from ROI and caption analysis."""
        subject = MainSubject()
        
        # Analyze ROI captions to determine subject
        roi_captions = [
            c.roi_caption for c in outputs.keyframe_captions
            if c.roi_caption
        ]
        
        if roi_captions:
            # Build hypothesis from most common terms
            hypothesis = self._synthesize_subject_hypothesis(roi_captions)
            subject.hypothesis = hypothesis
            
            # Extract appearance descriptors
            subject.appearance = self._extract_appearance(roi_captions)
            
            # Extract notable details
            subject.notable_details = self._extract_details(roi_captions)
            
            # Confidence based on caption consistency
            subject.confidence = self._compute_caption_confidence(roi_captions)
        else:
            subject.hypothesis = "unknown (no ROI captions available)"
            subject.confidence = 0.0
        
        # Add movement information if available
        if outputs.roi_movement:
            subject.notable_details.append(f"Movement: {outputs.roi_movement}")
        
        return subject
    
    def _synthesize_subject_hypothesis(
        self,
        roi_captions: List[str]
    ) -> str:
        """Synthesize best hypothesis for main subject from captions."""
        # Subject keywords to look for
        subject_terms = {
            "statue": 3,
            "sculpture": 3,
            "bust": 3,
            "figure": 2,
            "artifact": 2,
            "ruin": 2,
            "rock": 1,
            "coral": 1,
            "fish": 1,
            "object": 1,
            "structure": 1
        }
        
        term_counts = {}
        
        for caption in roi_captions:
            caption_lower = caption.lower()
            for term, weight in subject_terms.items():
                if term in caption_lower:
                    term_counts[term] = term_counts.get(term, 0) + weight
        
        if not term_counts:
            return "unidentified underwater object"
        
        # Get best term
        best_term = max(term_counts.items(), key=lambda x: x[1])[0]
        
        # Build hypothesis string
        if best_term in ["statue", "sculpture", "bust"]:
            return f"{best_term} (from ROI captions)"
        else:
            return f"possible {best_term} (from ROI captions)"
    
    def _extract_appearance(self, roi_captions: List[str]) -> List[str]:
        """Extract appearance descriptors from captions."""
        appearance_terms = [
            "weathered", "eroded", "ancient", "old", "covered",
            "encrusted", "algae", "sediment", "barnacles", "moss",
            "colorful", "pale", "dark", "bright", "faded",
            "partially buried", "submerged", "broken", "intact",
            "ornate", "simple", "detailed", "smooth", "rough"
        ]
        
        found = set()
        
        for caption in roi_captions:
            caption_lower = caption.lower()
            for term in appearance_terms:
                if term in caption_lower:
                    found.add(term)
        
        return list(found)[:6]  # Limit to 6 descriptors
    
    def _extract_details(self, roi_captions: List[str]) -> List[str]:
        """Extract notable details from captions."""
        detail_patterns = [
            ("hand", "visible hand/arm"),
            ("face", "facial features"),
            ("head", "head visible"),
            ("arm", "arm/limb visible"),
            ("base", "base/pedestal"),
            ("inscription", "possible inscription"),
            ("eyes", "eye detail"),
            ("nose", "facial profile"),
            ("drapery", "clothing/drapery"),
            ("hair", "hair detail")
        ]
        
        found = []
        
        for caption in roi_captions:
            caption_lower = caption.lower()
            for keyword, description in detail_patterns:
                if keyword in caption_lower and description not in found:
                    found.append(description)
        
        return found[:4]  # Limit to 4 details
    
    def _compute_caption_confidence(self, captions: List[str]) -> float:
        """
        Compute confidence score based on caption consistency.
        
        Higher score if captions agree on subject type.
        """
        if not captions:
            return 0.0
        
        # Count unique main subjects mentioned
        subjects_mentioned = set()
        for caption in captions:
            caption_lower = caption.lower()
            for subj in ["statue", "sculpture", "bust", "figure", "rock", "coral"]:
                if subj in caption_lower:
                    subjects_mentioned.add(subj)
        
        # More consistency = higher confidence
        if len(subjects_mentioned) == 0:
            return 0.2
        elif len(subjects_mentioned) == 1:
            return 0.8
        elif len(subjects_mentioned) == 2:
            return 0.5
        else:
            return 0.3
    
    def _process_motion_segments(
        self,
        segments: List[MotionSegment]
    ) -> List[MotionSegment]:
        """Process and clean up motion segments."""
        if not segments:
            return []
        
        # Filter out very short segments
        filtered = [s for s in segments if (s.t1 - s.t0) >= 0.2]
        
        # Merge adjacent segments with same labels
        merged = []
        for segment in filtered:
            if merged and merged[-1].labels == segment.labels:
                # Extend previous segment
                merged[-1] = MotionSegment(
                    t0=merged[-1].t0,
                    t1=segment.t1,
                    labels=segment.labels,
                    magnitude=(merged[-1].magnitude + segment.magnitude) / 2,
                    smoothness=(merged[-1].smoothness + segment.smoothness) / 2
                )
            else:
                merged.append(segment)
        
        return merged
    
    def _build_negatives(self, outputs: StageOutputs) -> NegativeEvidence:
        """Build negative evidence section."""
        return NegativeEvidence(
            people_detected=outputs.people_detected,
            fish_detected=outputs.fish_detected,
            text_detected=False,  # Could add text detection
            artificial_light=False  # Could detect from brightness patterns
        )
    
    def _select_best_keyframes(
        self,
        captions: List[KeyframeCaption],
        max_keyframes: int = 5
    ) -> List[KeyframeCaption]:
        """Select most informative keyframes for the report."""
        if not captions:
            return []
        
        # Score each keyframe by caption richness
        scored = []
        for caption in captions:
            score = 0
            if caption.global_caption:
                score += len(caption.global_caption.split()) * 0.5
            if caption.roi_caption:
                score += len(caption.roi_caption.split()) * 1.0  # ROI captions weighted higher
            scored.append((score, caption))
        
        # Sort by score and select top N
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [caption for _, caption in scored[:max_keyframes]]
        
        # Sort by timestamp for chronological order
        selected.sort(key=lambda c: c.timestamp)
        
        return selected
    
    def _generate_notes(self, outputs: StageOutputs) -> List[str]:
        """Generate processing notes for the report."""
        notes = []
        
        # Note about motion analysis
        if outputs.motion_segments:
            motion_types = set()
            for seg in outputs.motion_segments:
                motion_types.update(seg.labels)
            notes.append(f"Motion types detected: {', '.join(motion_types)}")
        
        # Note about ROI tracking
        if outputs.main_rois:
            notes.append(f"ROIs tracked: {len(outputs.main_rois)}")
        
        # Note about caption quality
        if outputs.keyframe_captions:
            valid_captions = sum(
                1 for c in outputs.keyframe_captions
                if c.global_caption or c.roi_caption
            )
            notes.append(f"Keyframes captioned: {valid_captions}")
        
        # Note about detection confidence
        if outputs.fish_detected is None:
            notes.append("Fish detection: uncertain (YOLO unavailable)")
        
        return notes
    
    def report_to_llm_context(self, report: SceneReport) -> str:
        """
        Convert scene report to a formatted string for LLM context.
        
        Alternative to JSON for prompting.
        """
        lines = [
            "=== SCENE ANALYSIS REPORT ===",
            "",
            f"Video: {report.video_path}",
            f"Duration: {report.duration_sec:.1f} seconds",
            "",
            "--- CONTEXT ---",
            f"Domain: {report.context.domain}",
            f"Platform: {report.context.platform}",
            f"Camera: {report.context.camera_style}",
            f"Constraints: {', '.join(report.context.constraints)}",
            "",
            "--- ENVIRONMENT ---",
            f"Water color: {report.environment.water_color}",
            f"Visibility: {report.environment.visibility}",
            f"Particulate: {report.environment.particulate}",
            f"Seafloor: {', '.join(report.environment.seafloor) or 'unknown'}",
            f"Lighting: {report.environment.lighting}",
            "",
            "--- MAIN SUBJECT ---",
            f"Hypothesis: {report.main_subject.hypothesis}",
            f"Appearance: {', '.join(report.main_subject.appearance) or 'unknown'}",
            f"Details: {', '.join(report.main_subject.notable_details) or 'none noted'}",
            f"Confidence: {report.main_subject.confidence:.0%}",
            "",
            "--- CAMERA MOTION ---"
        ]
        
        for seg in report.camera_motion:
            label_str = " + ".join(seg.labels)
            lines.append(f"  [{seg.t0:.1f}s - {seg.t1:.1f}s]: {label_str}")
        
        lines.extend([
            "",
            "--- NEGATIVE EVIDENCE ---",
            f"People detected: {report.negatives.people_detected}",
            f"Fish detected: {report.negatives.fish_detected}",
            "",
            "--- KEYFRAME CAPTIONS ---"
        ])
        
        for kf in report.keyframes:
            lines.append(f"  t={kf.timestamp:.1f}s:")
            if kf.global_caption:
                lines.append(f"    Global: {kf.global_caption}")
            if kf.roi_caption:
                lines.append(f"    ROI: {kf.roi_caption}")
        
        lines.extend([
            "",
            "--- PROCESSING NOTES ---"
        ])
        for note in report.processing_notes:
            lines.append(f"  • {note}")
        
        return "\n".join(lines)


def assemble_scene_report(
    outputs: StageOutputs,
    config: Optional[PipelineConfig] = None
) -> SceneReport:
    """
    Convenience function to assemble a scene report.
    """
    assembler = SceneReportAssembler(config)
    return assembler.assemble(outputs)
