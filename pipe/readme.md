# Retinal Diagnostic Pipeline

A module for automated retinal pathology diagnosis using deep learning models. This pipeline implements a two-stage diagnostic process:
1. Retinal Detachment (RD) Detection
2. Macula Status Classification (if RD is detected)

## Features
- Single video and batch processing modes
- Supports multiple model architectures (unet3d, swinunetr, etc.)
- Automatic checkpoint loading from experiment logs
- Easy-to-use CLI interface
- Flexible input handling (single video or CSV batch)

## Requirements
```bash
torch
torchvision
pandas
```

## Usage

### As a Python Module
```python
from src.diagnostic_pipeline import create_pipeline

# Initialize pipeline with your chosen model architecture
pipeline = create_pipeline(
    model_name="unet3d",  # or any other supported architecture
    experiment_root="logs"  # directory containing model checkpoints
)

# Single video prediction
result = pipeline.predict_single("path/to/video.mp4")
print(f"Has RD: {result['has_rd']}")
if result['has_rd']:
    print(f"Macula Detached: {result['macula_detached']}")

# Batch prediction from CSV
results_df = pipeline.predict_batch(
    csv_path="path/to/videos.csv",
    video_column="path"  # column containing video paths
)
```

### Command Line Interface
```bash
# Single video analysis
python -m pipe.diagnostic_pipeline \
    --model unet3d \
    --input path/to/video.mp4

# Batch processing
python -m pipe.diagnostic_pipeline \
    --model unet3d \
    --input path/to/videos.csv \
    --output pipe/results.csv
```

## Input Formats

### Single Video
- Supports .mp4 video files
- Video will be automatically preprocessed (resized, normalized)

### Batch Processing CSV
Example format:
```csv
path
data/video1.mp4
data/video2.mp4
```

## Output Format

### Single Video
```python
{
    'has_rd': bool,  # True if retinal detachment detected
    'macula_detached': Optional[bool]  # None if no RD, otherwise True/False
}
```

### Batch Processing
CSV file with added columns:
- `has_rd`: Boolean indicating RD detection
- `macula_detached`: Boolean/None indicating macula status

## Model Architecture Support
The pipeline supports all architectures implemented in the ERDES codebase:
- unet3d
- swinunetr
- unetplusplus
- vnet
- unetr
- resnet3d
- senet
- vit

## Directory Structure
The pipeline expects checkpoints to be organized as:
```
logs/
├── rd/
│   └── {model_name}/
│       └── checkpoints/
└── md/
    └── {model_name}/
        └── checkpoints/
```
