"""
Main Pipeline Orchestrator

Chains all stages together to process video and generate prompts:
1. OpenCV: sample frames → enhance
2. OpenCV: global motion estimation → motion segments
3. OpenCV: ROI propose + track (subject anchor)
4. Caption model: caption global + ROI on keyframes
5. Rules: build structured scene report JSON
6. LLM: generate final cinematic prompt + negative prompt
"""
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np

from config import PipelineConfig, get_default_config
from schemas import (
    SceneReport, LLMPromptOutput, PipelineOutput,
    MotionSegment, ROIData, KeyframeCaption
)

from stages.stage_a_sampling import VideoSampler, VideoMetadata
from stages.stage_b_preprocess import UnderwaterPreprocessor, EnhancedFrameResult
from stages.stage_c_motion import MotionAnalyzer, analyze_video_motion
from stages.stage_d_roi import ROIDiscovery
from stages.stage_e_caption import SemanticCaptioner, YOLODetector
from stages.stage_f_report import SceneReportAssembler, StageOutputs
from stages.stage_g_llm import LLMPromptSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    Complete video-to-prompt pipeline.
    
    Orchestrates all stages to process underwater ROV footage
    and generate cinematic prompts for video regeneration.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()
        
        # Initialize stage processors
        self.sampler = VideoSampler(self.config.sampling)
        self.preprocessor = UnderwaterPreprocessor(self.config.preprocess)
        self.motion_analyzer = MotionAnalyzer(self.config.motion)
        self.roi_discoverer = ROIDiscovery(self.config.roi)
        self.captioner = SemanticCaptioner(self.config.caption)
        self.report_assembler = SceneReportAssembler(self.config)
        self.llm_synthesizer = LLMPromptSynthesizer(
            self.config.llm, 
            self.config.context
        )
        
        # Optional YOLO detector
        self.yolo = YOLODetector()
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, video_path: str) -> PipelineOutput:
        """
        Process video through complete pipeline.
        
        Args:
            video_path: Path to input video file
            
        Returns: PipelineOutput with scene report and generated prompts
        """
        start_time = time.time()
        
        logger.info(f"Processing video: {video_path}")
        
        # Stage A: Sample frames
        logger.info("Stage A: Sampling frames...")
        metadata, samples = self._stage_a_sample(video_path)
        logger.info(f"  Sampled {len(samples)} frames from {metadata.duration_sec:.1f}s video")
        
        # Stage B: Preprocess frames
        logger.info("Stage B: Preprocessing frames...")
        enhanced_frames = self._stage_b_preprocess(samples)
        logger.info(f"  Enhanced {len(enhanced_frames)} frames")
        
        # Get environmental estimates from first few frames
        env_estimates = self._estimate_environment(enhanced_frames[:5])
        
        # Stage C: Analyze motion
        logger.info("Stage C: Analyzing motion...")
        motion_frames = [(ef.timestamp, ef.lowres) for ef in enhanced_frames]
        motion_segments = self._stage_c_motion(motion_frames)
        logger.info(f"  Found {len(motion_segments)} motion segments")
        
        # Stage D: Discover and track ROIs
        logger.info("Stage D: Discovering ROIs...")
        rois, roi_movement = self._stage_d_roi(enhanced_frames)
        logger.info(f"  Tracked {len(rois)} ROIs, movement: {roi_movement}")
        
        # Stage E: Caption keyframes
        logger.info("Stage E: Generating captions...")
        keyframes, captions = self._stage_e_caption(enhanced_frames, rois)
        logger.info(f"  Generated {len(captions)} keyframe captions")
        
        # Check for people/fish with YOLO (optional)
        people_detected, fish_detected = self._detect_objects(enhanced_frames)
        
        # Stage F: Assemble scene report
        logger.info("Stage F: Assembling scene report...")
        scene_report = self._stage_f_report(
            video_path=video_path,
            duration_sec=metadata.duration_sec,
            env_estimates=env_estimates,
            motion_segments=motion_segments,
            rois=rois,
            roi_movement=roi_movement,
            captions=captions,
            people_detected=people_detected,
            fish_detected=fish_detected
        )
        
        # Stage G: Synthesize prompt
        logger.info("Stage G: Synthesizing prompt...")
        llm_output = self._stage_g_llm(scene_report)
        
        processing_time = time.time() - start_time
        logger.info(f"Pipeline complete in {processing_time:.1f}s")
        
        # Build final output
        output = PipelineOutput(
            input_video=video_path,
            scene_report=scene_report,
            llm_output=llm_output,
            sampled_frame_count=len(samples),
            motion_segment_count=len(motion_segments),
            keyframe_count=len(captions),
            processing_time_sec=processing_time
        )
        
        # Save outputs
        if self.config.save_intermediate:
            self._save_outputs(output, video_path)
        
        return output
    
    def _stage_a_sample(
        self, 
        video_path: str
    ) -> Tuple[VideoMetadata, List[Tuple[float, int, np.ndarray]]]:
        """Stage A: Sample frames from video."""
        metadata = self.sampler.open(video_path)
        
        samples = []
        for sample, frame in self.sampler.sample_frames(return_frames=True):
            if frame is not None:
                samples.append((sample.timestamp, sample.frame_idx, frame))
        
        self.sampler.close()
        return metadata, samples
    
    def _stage_b_preprocess(
        self,
        samples: List[Tuple[float, int, np.ndarray]]
    ) -> List[EnhancedFrameResult]:
        """Stage B: Preprocess all sampled frames."""
        enhanced = []
        for timestamp, frame_idx, frame in samples:
            result = self.preprocessor.process_frame(frame, timestamp, frame_idx)
            enhanced.append(result)
        return enhanced
    
    def _estimate_environment(
        self,
        frames: List[EnhancedFrameResult]
    ) -> dict:
        """Estimate environmental conditions from frames."""
        if not frames:
            return {}
        
        # Use first frame for estimates
        first_frame = frames[0].enhanced
        
        return {
            "water_color": self.preprocessor.estimate_water_color(first_frame),
            "visibility": self.preprocessor.estimate_visibility(first_frame),
            "particulate": self.preprocessor.detect_particulate(first_frame)
        }
    
    def _stage_c_motion(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[MotionSegment]:
        """Stage C: Analyze motion between frames."""
        motions = self.motion_analyzer.analyze_sequence(frames)
        segments = self.motion_analyzer.segment_motion(motions)
        return segments
    
    def _stage_d_roi(
        self,
        frames: List[EnhancedFrameResult]
    ) -> Tuple[List[ROIData], Optional[str]]:
        """Stage D: Discover and track main subject ROI."""
        if not frames:
            return [], None
        
        # Discover ROI in first frame
        first_frame = frames[0].enhanced
        initial_rois = self.roi_discoverer.discover_rois(
            first_frame, frames[0].timestamp
        )
        
        if not initial_rois:
            return [], None
        
        # Track through sequence
        frame_sequence = [(ef.timestamp, ef.enhanced) for ef in frames]
        trajectory = self.roi_discoverer.track_sequence(
            frame_sequence, initial_rois[0]
        )
        
        return trajectory.rois, trajectory.movement_direction
    
    def _stage_e_caption(
        self,
        frames: List[EnhancedFrameResult],
        rois: List[ROIData]
    ) -> Tuple[List[np.ndarray], List[KeyframeCaption]]:
        """Stage E: Caption selected keyframes."""
        # Select keyframes evenly distributed
        n_keyframes = min(self.config.caption.frames_per_event, len(frames))
        if n_keyframes == 0:
            return [], []
        
        indices = np.linspace(0, len(frames) - 1, n_keyframes, dtype=int)
        
        keyframes = []
        captions = []
        
        for i in indices:
            frame = frames[i].enhanced
            timestamp = frames[i].timestamp
            keyframes.append(frame)
            
            # Find matching ROI
            matching_roi = None
            for roi in rois:
                if abs(roi.timestamp - timestamp) < 0.5:
                    matching_roi = roi
                    break
            
            caption = self.captioner.caption_frame_and_roi(
                frame, matching_roi, timestamp
            )
            captions.append(caption)
        
        return keyframes, captions
    
    def _detect_objects(
        self,
        frames: List[EnhancedFrameResult]
    ) -> Tuple[bool, Optional[bool]]:
        """Run YOLO detection on sample frames."""
        people_detected = False
        fish_detected = None
        
        # Check a few frames
        for frame in frames[::max(1, len(frames)//3)][:3]:
            detections = self.yolo.detect(frame.enhanced)
            
            if self.yolo.check_for_people(detections):
                people_detected = True
            
            fish_result = self.yolo.check_for_fish(detections)
            if fish_result is not None:
                if fish_detected is None:
                    fish_detected = fish_result
                elif fish_result:
                    fish_detected = True
        
        return people_detected, fish_detected
    
    def _stage_f_report(
        self,
        video_path: str,
        duration_sec: float,
        env_estimates: dict,
        motion_segments: List[MotionSegment],
        rois: List[ROIData],
        roi_movement: Optional[str],
        captions: List[KeyframeCaption],
        people_detected: bool,
        fish_detected: Optional[bool]
    ) -> SceneReport:
        """Stage F: Assemble structured scene report."""
        # Extract detected objects from captions
        detected_objects = []
        for caption in captions:
            objs = self.captioner.extract_objects_from_caption(
                caption.global_caption or ""
            )
            detected_objects.extend(objs)
            objs = self.captioner.extract_objects_from_caption(
                caption.roi_caption or ""
            )
            detected_objects.extend(objs)
        
        # Build stage outputs container
        outputs = StageOutputs(
            video_path=video_path,
            duration_sec=duration_sec,
            water_color=env_estimates.get("water_color"),
            visibility=env_estimates.get("visibility"),
            particulate=env_estimates.get("particulate"),
            motion_segments=motion_segments,
            main_rois=rois,
            roi_movement=roi_movement,
            keyframe_captions=captions,
            detected_objects=list(set(detected_objects)),
            people_detected=people_detected,
            fish_detected=fish_detected
        )
        
        return self.report_assembler.assemble(outputs)
    
    def _stage_g_llm(self, report: SceneReport) -> LLMPromptOutput:
        """Stage G: Synthesize final prompt using LLM."""
        return self.llm_synthesizer.synthesize(report)
    
    def _save_outputs(self, output: PipelineOutput, video_path: str):
        """Save pipeline outputs to files."""
        video_name = Path(video_path).stem
        
        # Save scene report
        report_path = self.output_dir / f"{video_name}_scene_report.json"
        with open(report_path, 'w') as f:
            json.dump(output.scene_report.model_dump(), f, indent=2, default=str)
        logger.info(f"Saved scene report: {report_path}")
        
        # Save LLM output
        prompt_path = self.output_dir / f"{video_name}_prompt.json"
        with open(prompt_path, 'w') as f:
            json.dump(output.llm_output.model_dump(), f, indent=2)
        logger.info(f"Saved prompt output: {prompt_path}")
        
        # Save full output
        full_path = self.output_dir / f"{video_name}_full_output.json"
        with open(full_path, 'w') as f:
            json.dump(output.model_dump(), f, indent=2, default=str)
        logger.info(f"Saved full output: {full_path}")
        
        # Save text-formatted report
        text_path = self.output_dir / f"{video_name}_report.txt"
        text_report = self.report_assembler.report_to_llm_context(output.scene_report)
        with open(text_path, 'w') as f:
            f.write(text_report)
            f.write("\n\n=== GENERATED PROMPT ===\n\n")
            f.write(f"Final Prompt:\n{output.llm_output.final_prompt}\n\n")
            f.write(f"Negative Prompt:\n{output.llm_output.negative_prompt}\n")


def process_video(
    video_path: str,
    config: Optional[PipelineConfig] = None
) -> PipelineOutput:
    """
    Convenience function to process a video.
    
    Args:
        video_path: Path to input video
        config: Optional pipeline configuration
        
    Returns: PipelineOutput with all results
    """
    pipeline = VideoPipeline(config)
    return pipeline.process(video_path)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Video-to-Prompt Pipeline for Underwater ROV Footage"
    )
    parser.add_argument(
        "video", 
        help="Path to input video file"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["minimax", "openai", "anthropic", "none"],
        default="minimax",
        help="LLM provider for prompt synthesis (default: minimax)"
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model (e.g., MiniMax-M2.1, gpt-4o, claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Skip image captioning (faster, less accurate)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only processing"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build config
    config = get_default_config()
    config.output_dir = Path(args.output)
    
    if args.llm_provider == "none":
        # Will use fallback synthesis
        config.llm.provider = "openai"
    else:
        config.llm.provider = args.llm_provider
    
    if args.llm_model:
        config.llm.model = args.llm_model
    
    if args.no_caption:
        config.caption.caption_global = False
        config.caption.caption_roi = False
    
    if args.cpu:
        config.caption.device = "cpu"
    
    # Run pipeline
    output = process_video(args.video, config)
    
    # Print results
    print("\n" + "="*60)
    print("GENERATED PROMPT")
    print("="*60)
    print(f"\n{output.llm_output.final_prompt}\n")
    
    print("NEGATIVE PROMPT")
    print("-"*40)
    print(f"{output.llm_output.negative_prompt}\n")
    
    print("CONFIDENCE NOTES")
    print("-"*40)
    for note in output.llm_output.confidence_notes:
        print(f"  • {note}")
    
    print(f"\nProcessing time: {output.processing_time_sec:.1f}s")
    print(f"Outputs saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
