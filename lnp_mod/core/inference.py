"""
LNP-MOD Inference Module

This module provides the main inference pipeline for LNP detection and segmentation.
"""

import os
import argparse
import torch
import cv2
from typing import Optional, List, Tuple
from tqdm import tqdm

import lnp_mod.utils.utils as utils
import lnp_mod.config.constants as constants
from lnp_mod.core.post_process import post_process
from lnp_mod.utils.load_model import load_model_files
from lnp_mod.core.predict_sam import inference_with_sam, inference_with_sam2, object_detection_inference


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="LNP-MOD: Lipid Nanoparticle Morphology Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('-i', '--input_dir', 
                       help='Input directory path containing images', 
                       required=True)
    
    # Optional arguments
    parser.add_argument('-o', '--output_dir', 
                       help='Output directory path', 
                       default=None)
    parser.add_argument('--conf', 
                       help='Confidence threshold for object detection', 
                       type=float, default=0.25)
    parser.add_argument('--iou', 
                       help='IOU threshold for object detection', 
                       type=float, default=0.8)
    parser.add_argument('--imgsz', 
                       help='Image size for object detection', 
                       type=int, default=2048)
    parser.add_argument('--od_model', 
                       help='Object detection model to use', 
                       choices=list(constants.OD_MODEL_FILES.keys()),
                       default='yolov12')
    parser.add_argument('--seg_model',
                       help='Segmentation model to use',
                       choices=list(constants.SEG_MODEL_FILES.keys()),
                       default='sam2.1_hiera_base_plus_EP40')
    parser.add_argument('--simplify', 
                       help='Apply polygon simplification', 
                       action='store_true', default=True)
    parser.add_argument('--epsilon', 
                       help='Polygon simplification level (higher = fewer points)', 
                       type=float, default=1.0)
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Check for supported images
    supported_images = utils.get_supported_images_path_list(args.input_dir)
    if len(supported_images) == 0:
        raise ValueError(f"No supported images found in input directory: {args.input_dir}")
    
    # Validate model selections
    if args.od_model not in constants.OD_MODEL_FILES:
        available_models = list(constants.OD_MODEL_FILES.keys())
        raise ValueError(f"Invalid object detection model: {args.od_model}. "
                        f"Available models: {available_models}")
    
    if args.seg_model not in constants.SEG_MODEL_FILES:
        available_models = list(constants.SEG_MODEL_FILES.keys())
        raise ValueError(f"Invalid segmentation model: {args.seg_model}. "
                        f"Available models: {available_models}")


def setup_output_directory(input_dir: str, output_dir: Optional[str] = None) -> str:
    """Set up output directory."""
    if output_dir is None:
        output_dir = utils.create_output_folder(input_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    return os.path.abspath(output_dir)


def load_models(od_model_name: str, seg_model_name: str, model_dir: str) -> Tuple[str, str]:
    """Load object detection and segmentation models."""
    print(f"Loading object detection model: {od_model_name}")
    od_model = load_model_files(
        model_dir,
        constants.OD_MODEL_FILES[od_model_name]
    )
    
    print(f"Loading segmentation model: {seg_model_name}")
    segmentation_model = load_model_files(
        model_dir,
        constants.SEG_MODEL_FILES[seg_model_name]
    )
    
    return od_model, segmentation_model


def preprocess_images(input_images_paths: List[str], 
                     output_dir: str) -> Tuple[str, List[str]]:
    """Preprocess images."""
    return os.path.dirname(input_images_paths[0]), input_images_paths


def run_inference_pipeline(input_dir: str,
                          output_dir: Optional[str] = None,
                          od_model: str = 'yolov12',
                          seg_model: str = 'sam2.1_hiera_base_plus_EP40',
                          conf: float = 0.25,
                          iou: float = 0.8,
                          imgsz: int = 2048,
                          simplify: bool = True,
                          epsilon: float = 1.0) -> str:
    """
    Run the complete LNP inference pipeline.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results (created automatically if None)
        od_model: Object detection model name
        seg_model: Segmentation model name
        conf: Confidence threshold for object detection
        iou: IOU threshold for object detection
        imgsz: Image size for object detection
        simplify: Whether to apply polygon simplification
        epsilon: Polygon simplification parameter
        
    Returns:
        Path to the output JSON file with results
        
    Raises:
        ValueError: If input validation fails
        FileNotFoundError: If required files are not found
    """
    # Setup and validation
    input_dir = os.path.abspath(input_dir)
    output_dir = setup_output_directory(input_dir, output_dir)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get input images
    utils.disable_jpg_images(input_dir)  # Disable JPG overview images
    input_images_paths = utils.get_supported_images_path_list(input_dir)
    
    if len(input_images_paths) == 0:
        raise ValueError(f"No supported images found in {input_dir}")
    
    print(f"Found {len(input_images_paths)} images for processing")
    
    # Load models
    model_dir = os.path.join(os.getcwd(), 'models')
    od_model_path, segmentation_model_path = load_models(od_model, seg_model, model_dir)
    
    # Preprocess images
    image_dir, source_paths = preprocess_images(input_images_paths, output_dir)
    
    try:
        # Run object detection
        print("Running object detection...")
        od_result_json = object_detection_inference(
            od_model_path, source_paths, output_dir, conf, iou, imgsz
        )
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run segmentation
        print("Running segmentation...")
        pipeline_result = inference_with_sam2(
            segmentation_model_path,
            image_dir,
            output_dir,
            od_result_json,
            simplify_polygons=simplify,
            epsilon=epsilon
        )
        
        # Post-process results
        print("Post-processing results...")
        output_json_path = os.path.join(output_dir, 'lnp_mod_output.json')
        post_process(output_json_path, image_dir, output_dir)
        
        print(f"Pipeline complete! Results saved to: {output_dir}")
        return output_json_path
        
    finally:
        # Always restore JPG images
        utils.restore_jpg_images(input_dir)


def main() -> None:
    """Main entry point for command-line interface."""
    try:
        # Parse arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Validate arguments
        validate_arguments(args)
        
        # Run inference pipeline
        result_path = run_inference_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            od_model=args.od_model,
            seg_model=args.seg_model,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            simplify=args.simplify,
            epsilon=args.epsilon
        )
        
        print(f"✅ Inference completed successfully!")
        print(f"📁 Results available at: {os.path.dirname(result_path)}")
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())