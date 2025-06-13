"""
LNP-MOD Core Module

This module contains the core functionality for LNP detection and segmentation.
"""

from .predict_sam import (
    object_detection_inference,
    inference_with_sam,
    inference_with_sam2
)
from .post_process import post_process
from .nms import NMSProcessor

# Import inference functions but handle potential import issues
try:
    from .inference import run_inference_pipeline, main as inference_main
except ImportError:
    # Handle gracefully if there are import issues
    run_inference_pipeline = None
    inference_main = None

__all__ = [
    'run_inference_pipeline',
    'inference_main',
    'object_detection_inference', 
    'inference_with_sam',
    'inference_with_sam2',
    'post_process',
    'NMSProcessor'
]
