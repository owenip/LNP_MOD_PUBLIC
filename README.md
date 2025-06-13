# LNP-MOD

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/owenip/LNP_MOD_PUBLIC/blob/master/colab_notebooks/LNP-MOD.ipynb)
[![Training Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/owenip/LNP_MOD_PUBLIC/blob/master/colab_notebooks/YoloV8Training.ipynb)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**LNP-MOD** (Lipid Nanoparticle Morphology Detection) is a training framework and computer vision pipeline for automatically detecting, segmenting, and analyzing lipid nanoparticles (LNPs) in electron microscopy images. This repository provides the training framework and inference pipeline using state-of-the-art object detection (YOLO) and segmentation (SAM) models.

## Overview

LNP-MOD provides a complete pipeline for:
- **Detection**: Identifying different types of lipid nanoparticles using YOLO models
- **Segmentation**: Precise boundary delineation using Segment Anything Model (SAM)
- **Analysis**: Quantitative characterization of particle morphology and mRNA content
- **Visualization**: Annotated images and statistical reports

### Key Features

- Training framework for custom YOLO detection models
- SAM model fine-tuning pipeline for precise segmentation
- Automatic mRNA detection within LNPs
- Morphological analysis (size, shape, containment relationships)
- Batch processing capabilities
- Export results in multiple formats (Excel, COCO, annotated images)

### Supported LNP Types

- Bleb with mRNA
- mRNA particles
- Oil Core LNPs
- Liposomal LNPs
- Oil Droplets
- Partially visible LNPs
- Other LNP morphologies

## Quick Start

**Note**: This repository provides a training framework. You will need to train your own models before using the inference pipeline.

### Google Colab

You can explore the training framework through Google Colab:

1. Click the "Open in Colab" button above
2. Follow the training notebook instructions
3. Train your models using the provided framework

### Local Installation

1. Clone the repository

```bash
git clone https://github.com/owenip/LNP_MOD_PUBLIC
cd LNP-MOD
```

2. Set up the environment

```bash
# Using conda (recommended)
conda env create -f environment.yaml
conda activate lnp_mod

# Or using pip
pip install -r requirements.txt
```

3. Train your models

```bash
# Follow the training documentation to:
# 1. Prepare your dataset
# 2. Train detection models
# 3. Fine-tune segmentation models
# 4. Place trained models in models/ directory
# See colab_notebooks/ and tools/ for training resources
```

## Usage

### Command Line Interface

```bash
# Using the module directly
python -m lnp_mod.core.inference --input_dir <path_to_input_dir> --output_dir <path_to_output_dir>

# Or using the installed command (after pip install)
lnp-mod --input_dir <path_to_input_dir> --output_dir <path_to_output_dir>

```

### Programmatic API

```python
from lnp_mod.core.inference import run_inference_pipeline

# Basic usage
result_path = run_inference_pipeline(input_dir="path/to/images")

# Custom parameters
result_path = run_inference_pipeline(
    input_dir="path/to/images",
    output_dir="path/to/output",
    od_model='yolov12',
    conf=0.3
)
```

### Available Parameters

- `input_dir`: Directory containing input images (required)
- `output_dir`: Output directory (auto-generated if not specified)
- `od_model`: Object detection model (configure in constants.py)
- `seg_model`: Segmentation model (configure in constants.py)
- `conf`: Confidence threshold (default: 0.25)
- `iou`: IOU threshold (default: 0.8)
- `imgsz`: Image size for processing (default: 2048)
- `simplify`: Apply polygon simplification (default: True)
- `epsilon`: Polygon simplification level (default: 1.0)

## Package Structure

```
lnp_mod/
├── __init__.py              # Package initialization
├── config/
│   ├── __init__.py
│   └── constants.py         # Model URLs, categories, and physical constants
├── core/
│   ├── __init__.py
│   ├── inference.py         # Main inference pipeline and CLI
│   ├── nms.py              # Non-maximum suppression utilities
│   ├── post_process.py     # Analysis and quantification
│   └── predict_sam.py      # SAM segmentation implementation
├── custom_yolov8/
│   ├── __init__.py
│   ├── augment.py          # Data augmentation utilities
│   └── trainer.py          # Custom YOLO training
└── utils/
    ├── __init__.py
    ├── image_processing/
    │   ├── __init__.py
    ├── load_model.py       # Model loading utilities
    └── utils.py            # General utilities

```

## Core Modules

### `config.constants`
Defines all system constants including:
- Model configuration for your trained models
- Category mappings for detection, segmentation, and post-processing
- Physical constants (pixel-to-nm conversion, mRNA size metrics)
- Visualization color schemes

### `core.inference`
Main entry point providing:
- Command-line interface with comprehensive arguments
- Pipeline orchestration (detection → segmentation → post-processing)
- Model loading and management
- Batch image processing

### `core.predict_sam`
Implements the two-stage detection and segmentation:
- YOLO-based object detection for initial bounding boxes
- SAM-based segmentation for precise masks
- Custom NMS for overlapping detections
- COCO format export

### `core.post_process`
Performs quantitative analysis:
- Spatial relationship analysis between particles
- mRNA containment detection
- Morphological measurements (area, dimensions)
- Statistical report generation (Excel, plots)

### `utils.image_processing`
Preprocessing utilities:
- Image format conversions
- Batch processing helpers

## Model Training

This repository provides a complete training framework:

### 1. Dataset Preparation
- Convert your annotated data to COCO format
- Use tools in `tools/` directory for dataset conversion
- See `tools/coco2yolo.py` and related scripts

### 2. Object Detection Training
- Use `lnp_mod/custom_yolov8/trainer.py` for YOLO training
- Training notebook: `colab_notebooks/YoloV8Training.ipynb`
- Supports YOLOv8/v9/v10/v11/v12 architectures

### 3. Segmentation Model Fine-tuning
- Fine-tune SAM models for your specific domain
- Use `tools/convert_sam2_2pt.py` for SAM2 conversion
- Custom augmentations available in `albumentations/`

### 4. Model Configuration
- Place trained models in `models/` directory
- Update `lnp_mod/config/constants.py` with your model paths
- Configure category mappings for your classes

### Training Resources
- Training notebooks in `colab_notebooks/`
- Utility scripts in `tools/`
- Custom augmentation pipeline in `albumentations/`
- Dataset conversion tools

## Output Files

After processing, LNP-MOD generates:

1. **Annotated Images** (`annotated/`)
   - Original images with detection boxes and segmentation masks
   - Color-coded by particle type

2. **Quantitative Analysis** (`post_process/`)
   - Excel file with detailed measurements
   - Statistical plots (area distributions, morphology analysis)
   - Summary statistics

3. **Raw Results**
   - COCO format JSON files
   - Individual mask files
   - Detection confidence scores


## Technical Details

### Physical Constants

- Pixel to nanometer conversion: 4.68 nm/pixel
- mRNA reference size: 208.44 nm² (area), 1999 nm³ (volume)
- Default image processing size: 2048×2048 pixels

### Processing Pipeline

1. **Image Loading**: Supports common formats (PNG, JPG, TIF)
2. **Detection**: YOLO model identifies particle bounding boxes
3. **Segmentation**: SAM refines boxes to precise masks
4. **Post-processing**: Analyzes relationships and morphology
5. **Export**: Saves results in multiple formats

## Citation

If you use LNP-MOD in your research, please cite:

```bibtex
[Citation information to be added]
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

**You are free to:**
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.

For commercial use, please contact the authors for licensing arrangements.
