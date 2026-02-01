"""
Pipeline stages for video-to-prompt processing.
"""
from .stage_a_sampling import VideoSampler
from .stage_b_preprocess import UnderwaterPreprocessor
from .stage_c_motion import MotionAnalyzer
from .stage_d_roi import ROIDiscovery
from .stage_e_caption import SemanticCaptioner
from .stage_f_report import SceneReportAssembler
from .stage_g_llm import LLMPromptSynthesizer

__all__ = [
    "VideoSampler",
    "UnderwaterPreprocessor", 
    "MotionAnalyzer",
    "ROIDiscovery",
    "SemanticCaptioner",
    "SceneReportAssembler",
    "LLMPromptSynthesizer",
]
