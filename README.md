# Video-to-Prompt Pipeline for Underwater ROV Footage

> **📡 This is the TRANSMITTER part of the Semantic Underwater Communication system.**  
> For the Receiver part, see: [Semantic_Underwater_communication_Receiver](https://github.com/soham12112/Semantic_Underwater_communication_Receiver)

A model-chained architecture that converts underwater ROV video into structured scene reports and cinematic generation prompts. Designed to run mostly on CPU with OpenCV, using an LLM only for final prompt synthesis.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VIDEO-TO-PROMPT PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Stage A  │───▶│ Stage B  │───▶│ Stage C  │───▶│ Stage D  │          │
│  │ Sampling │    │ Preproc  │    │ Motion   │    │   ROI    │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       │              │               │               │                  │
│       └──────────────┴───────────────┴───────────────┘                  │
│                            │                                            │
│                            ▼                                            │
│                     ┌──────────┐                                        │
│                     │ Stage E  │                                        │
│                     │ Caption  │                                        │
│                     └──────────┘                                        │
│                            │                                            │
│                            ▼                                            │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │                    Stage F: Scene Report                     │      │
│   │  ┌─────────────────────────────────────────────────────┐    │      │
│   │  │ { context, environment, main_subject, camera_motion, │    │      │
│   │  │   negatives, keyframes, processing_notes }            │    │      │
│   │  └─────────────────────────────────────────────────────┘    │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            │                                            │
│                            ▼                                            │
│                     ┌──────────┐                                        │
│                     │ Stage G  │                                        │
│                     │   LLM    │                                        │
│                     └──────────┘                                        │
│                            │                                            │
│                            ▼                                            │
│            ┌────────────────────────────────┐                           │
│            │ final_prompt + negative_prompt │                           │
│            └────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Stages

### Stage A — Video Sampling (OpenCV)
- Samples at 1-2 fps for scene understanding
- Optional burst sampling at 5-8 fps around detected events

### Stage B — Underwater Preprocessing (OpenCV)
- Gray-world white balance
- CLAHE contrast enhancement
- Optional mild dehazing
- Downscaling for consistency

### Stage C — Motion Analysis (OpenCV)
- ORB + RANSAC homography estimation
- Classifies motion types:
  - **truck left/right** (lateral movement)
  - **pedestal up/down** (vertical movement)
  - **push in/pull out** (zoom/dolly)
  - **static hold**

### Stage D — ROI Discovery & Tracking (OpenCV)
- Edge density / saliency-based ROI detection
- CSRT tracker for temporal consistency
- Works even when YOLO fails on unusual subjects

### Stage E — Lightweight Semantics
- BLIP-2 captioning on global frames and ROI crops
- Optional YOLO for negative evidence (people/fish detection)

### Stage F — Scene Report Assembly (Rule-Based)
- Consolidates all signals into structured JSON
- Resolves conflicts, deduplicates information
- Prepares evidence for LLM consumption

### Stage G — LLM Prompt Synthesis
- Converts scene report to cinematic prompt
- Generates main prompt + negative prompt
- Supports OpenAI and Anthropic APIs
- Fallback rule-based synthesis when LLM unavailable

## Installation

```bash
# Clone the repository
cd vid_to_txt_2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: GPU Support
```bash
# For CUDA-accelerated captioning
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### API Keys
Set your API key for LLM synthesis:

```bash
# For MiniMax M2.1 (recommended - best instruction following)
# Get your key from: https://platform.minimax.io
export MINIMAX_API_KEY="your-key-here"
# Optional: export MINIMAX_GROUP_ID="your-group-id"

# For OpenAI (alternative)
export OPENAI_API_KEY="your-key-here"

# For Anthropic (alternative)
export ANTHROPIC_API_KEY="your-key-here"
```

**Windows PowerShell:**
```powershell
$env:MINIMAX_API_KEY="your-key-here"
```

## Usage

### Command Line
```bash
# Basic usage (uses MiniMax M2.1 by default)
python pipeline.py video.mp4

# With options
python pipeline.py video.mp4 \
    --output ./results \
    --llm-provider minimax \
    --llm-model MiniMax-M2.1 \
    --verbose

# Use MiniMax M2.1-lightning (faster)
python pipeline.py video.mp4 --llm-model MiniMax-M2.1-lightning

# CPU-only, skip captioning (fastest)
python pipeline.py video.mp4 --cpu --no-caption

# Use OpenAI instead
python pipeline.py video.mp4 --llm-provider openai --llm-model gpt-4o

# Use Anthropic instead
python pipeline.py video.mp4 --llm-provider anthropic --llm-model claude-3-5-sonnet-20241022
```

### Python API
```python
from pipeline import VideoPipeline, process_video
from config import PipelineConfig

# Simple usage (uses MiniMax M2.1 by default)
output = process_video("video.mp4")
print(output.llm_output.final_prompt)
print(output.llm_output.negative_prompt)

# With custom config
config = PipelineConfig()
config.sampling.base_fps = 2.0
config.llm.provider = "minimax"
config.llm.model = "MiniMax-M2.1"  # or "MiniMax-M2.1-lightning" for faster

pipeline = VideoPipeline(config)
output = pipeline.process("video.mp4")

# Access scene report
print(output.scene_report.main_subject.hypothesis)
print(output.scene_report.camera_motion)

# Use OpenAI instead
config.llm.provider = "openai"
config.llm.model = "gpt-4o"
```

## Output Format

### Scene Report JSON
```json
{
  "video_path": "video.mp4",
  "duration_sec": 4.2,
  "context": {
    "domain": "underwater",
    "platform": "ROV",
    "camera_style": "documentary GoPro",
    "constraints": ["realistic colors", "smooth motion"]
  },
  "environment": {
    "water_color": "turquoise-green",
    "visibility": "slight haze",
    "particulate": "visible",
    "seafloor": ["pale sand", "coral rubble"]
  },
  "main_subject": {
    "hypothesis": "statue bust (from ROI captions)",
    "appearance": ["weathered", "algae-covered"],
    "notable_details": ["facial features visible"],
    "confidence": 0.8
  },
  "camera_motion": [
    {"t0": 0.0, "t1": 1.8, "labels": ["push_in", "pedestal_down"]},
    {"t0": 1.8, "t1": 2.5, "labels": ["static_hold"]},
    {"t0": 2.5, "t1": 4.0, "labels": ["truck_right"]}
  ],
  "negatives": {
    "people_detected": false,
    "fish_detected": null
  },
  "keyframes": [...]
}
```

### LLM Output JSON
```json
{
  "final_prompt": "Underwater ROV footage pushing in toward a weathered stone statue bust partially buried in pale sand, turquoise-green water with slight haze and visible particulate, the camera holds briefly then trucks right revealing coral rubble, documentary GoPro wide-angle style with natural lighting.",
  "negative_prompt": "people, divers, fish, marine life, blurry, shaky, artificial lighting, CGI",
  "confidence_notes": ["Subject identification based on consistent ROI captions"],
  "used_evidence": ["motion_segments: 3", "subject_confidence: 80%"],
  "suggested_duration_sec": 3.0,
  "style_tags": ["documentary", "underwater", "ROV", "GoPro"]
}
```

## Configuration

Edit `config.py` or pass a `PipelineConfig` object:

```python
from config import PipelineConfig, SamplingConfig, LLMConfig

config = PipelineConfig(
    sampling=SamplingConfig(
        base_fps=1.5,
        burst_fps=6.0
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.3
    )
)
```

### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `sampling.base_fps` | 1.5 | Base frame sampling rate |
| `preprocess.clahe_clip_limit` | 2.0 | CLAHE contrast limit |
| `motion.translation_threshold` | 15.0 | Pixels for motion detection |
| `roi.method` | "saliency" | ROI detection method |
| `caption.model_name` | "Salesforce/blip2-opt-2.7b" | Captioning model |
| `llm.provider` | "minimax" | LLM API provider (minimax/openai/anthropic) |
| `llm.model` | "MiniMax-M2.1" | LLM model name |

### LLM Provider Options

| Provider | Model Options | Notes |
|----------|--------------|-------|
| `minimax` | `MiniMax-M2.1`, `MiniMax-M2.1-lightning`, `MiniMax-M2` | **Recommended** - Best instruction following |
| `openai` | `gpt-4o`, `gpt-4-turbo` | Good alternative |
| `anthropic` | `claude-3-5-sonnet-20241022` | Good alternative |

MiniMax M2.1 is recommended because of its [exceptional instruction following and structured output capabilities](https://www.minimax.io/news/minimax-m21).

## Project Structure

```
vid_to_txt_2/
├── config.py              # Configuration dataclasses
├── schemas.py             # Pydantic models for data structures
├── pipeline.py            # Main orchestrator + CLI
├── requirements.txt       # Dependencies
├── README.md              # This file
└── stages/
    ├── __init__.py
    ├── stage_a_sampling.py    # Video ingestion
    ├── stage_b_preprocess.py  # Underwater enhancement
    ├── stage_c_motion.py      # Motion analysis
    ├── stage_d_roi.py         # ROI discovery/tracking
    ├── stage_e_caption.py     # Image captioning
    ├── stage_f_report.py      # Scene report assembly
    └── stage_g_llm.py         # LLM prompt synthesis
```

## Performance

Typical processing times (4-second video):

| Configuration | Time |
|---------------|------|
| Full pipeline (CPU) | ~30-60s |
| Full pipeline (GPU) | ~15-25s |
| Without captioning | ~5-10s |
| Without LLM | ~25-50s |

## Extending the Pipeline

### Custom Context Template
```python
from config import ContextTemplate

context = ContextTemplate(
    domain="underwater",
    platform="AUV",  # Autonomous Underwater Vehicle
    camera_style="scientific survey camera",
    priorities=["accuracy", "detail", "consistency"],
    forbidden=["artistic interpretation", "color enhancement"]
)
```

### Custom ROI Detection
```python
from stages.stage_d_roi import ROIDiscovery

class CustomROIDiscovery(ROIDiscovery):
    def discover_rois(self, frame, timestamp):
        # Your custom detection logic
        pass
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision primitives
- Hugging Face Transformers for BLIP-2 captioning
- [MiniMax M2.1](https://www.minimax.io/news/minimax-m21) for excellent LLM prompt synthesis
- OpenAI/Anthropic for alternative LLM APIs
