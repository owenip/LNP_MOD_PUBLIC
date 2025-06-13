# Models Directory

This directory is where you should place your trained model files.

## Required Models

To use LNP-MOD, you need to provide the following trained models:

### Object Detection Models
- **Format**: PyTorch `.pt` files trained with YOLOv8/v9/v10/v11/v12
- **Purpose**: Initial detection of LNP bounding boxes
- **Configure in**: `lnp_mod/config/constants.py` under `OD_MODEL_FILES`

### Segmentation Models  
- **Format**: PyTorch `.pt` files (SAM-based models)
- **Purpose**: Precise segmentation masks from detection boxes
- **Configure in**: `lnp_mod/config/constants.py` under `SEG_MODEL_FILES`

## Training Your Models

### Object Detection Training
Use the custom YOLO trainer provided in:
- `lnp_mod/custom_yolov8/trainer.py`
- Training notebook: `colab_notebooks/YoloV8Training.ipynb`

### Segmentation Model Fine-tuning
Use the SAM fine-tuning tools:
- `tools/convert_sam2_2pt.py` - Convert SAM2 checkpoints
- See training documentation for SAM fine-tuning procedures

## Model Configuration

After training, update `lnp_mod/config/constants.py`:

```python
OD_MODEL_FILES = {
    'your_model_name': {
        'url': 'local',
        'filename': 'your_detection_model.pt'
    }
}

SEG_MODEL_FILES = {
    'your_seg_model': {
        'url': 'local', 
        'filename': 'your_segmentation_model.pt'
    }
}
```

## File Structure

```
models/
├── README.md           # This file
├── .gitignore         # Excludes .pt files from git
├── .gitkeep          # Keeps directory in git
└── your_models.pt    # Your trained model files (place here)
```