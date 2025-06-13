#!/usr/bin/env python
import os
import torch
import argparse
from segment_anything import sam_model_registry

def convert_sam_checkpoint(checkpoint_path, output_path):
    """
    Convert SAM official checkpoint to a torch model that can be loaded with torch.load()
    
    Args:
        checkpoint_path: Path to the official SAM checkpoint (.pth file)
        output_path: Path to save the converted model (.pt file)
    """
    print(f"Loading SAM checkpoint from {checkpoint_path}")
    
    # Determine model type based on filename (b=base, l=large, h=huge)
    if "vit_b" in os.path.basename(checkpoint_path):
        model_type = "vit_b"
    elif "vit_l" in os.path.basename(checkpoint_path):
        model_type = "vit_l"
    elif "vit_h" in os.path.basename(checkpoint_path):
        model_type = "vit_h"
    else:
        raise ValueError(f"Could not determine model type from checkpoint: {checkpoint_path}")
    
    print(f"Detected model type: {model_type}")
    
    # Load the model using the SAM model registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.eval()  # Set to evaluation mode
    
    # Save the model with torch.save
    print(f"Saving converted model to {output_path}")
    torch.save(sam, output_path)
    
    print("Conversion complete!")
    print(f"You can now use this model with inference.py and predict_sam.py")
    print(f"Add the following to lnp_mod/config/constants.py SEG_MODEL_FILES:")
    model_name = os.path.basename(output_path).split('.')[0]
    print(f"    '{model_name}': '{os.path.basename(output_path)}',")

def main():
    parser = argparse.ArgumentParser(description='Convert SAM checkpoint to PyTorch model')
    parser.add_argument('--checkpoint', required=True, help='Path to SAM checkpoint (.pth file)')
    parser.add_argument('--output', help='Path to save converted model (.pt file)')
    
    args = parser.parse_args()
    
    # If output path not provided, create it from the checkpoint name
    if args.output is None:
        checkpoint_basename = os.path.basename(args.checkpoint)
        model_name = os.path.splitext(checkpoint_basename)[0]
        args.output = f"{model_name}_converted.pt"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    convert_sam_checkpoint(args.checkpoint, args.output)

if __name__ == "__main__":
    main() 