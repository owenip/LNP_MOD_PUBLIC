import torch
from sam2.build_sam import build_sam2

def build_sam2_checkpoint(checkpoint_path, model_type="base+", device="cpu"):
    """
    Build SAM2 model and save checkpoint for inference
    Args:
        checkpoint_path: Path to save the checkpoint
        model_type: SAM2 model type ("base+", "large", "small", "tiny")
        device: Device to load model on ("cuda" or "cpu")
    """
    # Map model type to correct config and checkpoint names
    config_map = {
        "base+": "sam2.1_hiera_b+",
        "large": "sam2.1_hiera_l",
        "small": "sam2.1_hiera_s",
        "tiny": "sam2.1_hiera_t"
    }
    
    checkpoint_map = {
        "base+": "sam2.1_hiera_base_plus",
        "large": "sam2.1_hiera_large",
        "small": "sam2.1_hiera_small",
        "tiny": "sam2.1_hiera_tiny"
    }
    
    config_name = config_map.get(model_type)
    checkpoint_name = checkpoint_map.get(model_type)
    
    if not config_name or not checkpoint_name:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(config_map.keys())}")
    
    # Use the correct path format for Hydra
    config_path = f"configs/sam2.1/{config_name}"
    
    # Get path to checkpoints - UPDATE THIS PATH to your SAM2 installation
    checkpoint_file = f"path/to/sam2/checkpoints/{checkpoint_name}.pt"
    
    # Initialize SAM2 model
    model = build_sam2(
        config_file=config_path,
        ckpt_path='path/to/sam2/checkpoints/sam2.1_base_plus_EP40.pt',  # UPDATE THIS PATH
        device=device,
        mode="eval"
    )

    # Save the entire model (not just state_dict)
    torch.save(model, checkpoint_path)
    print(f"SAM2 model saved to: {checkpoint_path}")
    print(f"You can load it with: model = torch.load('{checkpoint_path}')")
    print(f"And use it directly with: model.eval()")

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "models/finetuned_sam2_base_plus.pt"
    build_sam2_checkpoint(checkpoint_path)


