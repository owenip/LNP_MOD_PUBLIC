import json
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import lnp_mod.config.constants as constants
import torch
import supervision as sv
from tqdm import tqdm
from ultralytics.data.utils import LOGGER
from .nms import NMSProcessor

from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from sam2.utils.transforms import SAM2Transforms
from pycocotools.coco import COCO
from lnp_mod.utils.utils import mask_to_single_polygon
import copy
import tempfile


def object_detection_inference(model, input_images_paths, output_dir, conf=0.6, iou=0.7, imgsz=2048):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    od_model = YOLO(model)
    nms_processor = NMSProcessor(
        mrna_iou_threshold=constants.SAM_MRNA_IOU_THRESHOLD,
        mrna_size_ratio=constants.SAM_MRNA_SIZE_RATIO,
        mrna_overlap_threshold=constants.SAM_MRNA_OVERLAP_THRESHOLD,
        mrna_containment_threshold=constants.SAM_MRNA_CONTAINMENT_THRESHOLD
    )

    categories = []
    for cls, cls_name in constants.SEGMENTATION_CATEGORIES.items():
        categories.append({
            "id": cls,
            "name": cls_name
        })

    new_coco_data = {
        'images': [],
        'annotations': [],
        'categories': categories,
    }

    for image_path in input_images_paths:
        # Ensure we have an absolute path
        abs_image_path = os.path.abspath(image_path)
        
        # Check if file exists
        if not os.path.isfile(abs_image_path):
            print(f"Warning: Image file not found: {abs_image_path}")
            continue
            
        detection_results = od_model.predict(
            source=abs_image_path,
            device=device,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
        )

        img_id = len(new_coco_data['images']) + 1
        img_name = os.path.basename(image_path)
        image = cv2.imread(abs_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        new_coco_data['images'].append({
            "id": img_id,
            "file_name": img_name,
            "width": image.shape[1],
            "height": image.shape[0],
        })

        for i in range(len(detection_results)):
            result = detection_results[i].cpu().numpy()
            
            xyxy_boxes = result.boxes.xyxy
            cls = result.boxes.cls
            confs = result.boxes.conf
            
            # Apply custom NMS
            keep_indices = nms_processor._custom_nms(
                boxes=xyxy_boxes,
                scores=confs,
                class_ids=cls
            )
            
            # Filter boxes based on NMS results
            xyxy_boxes = xyxy_boxes[keep_indices]
            cls = cls[keep_indices]
            confs = confs[keep_indices]
            
            # Convert to xywh format for COCO
            xywh_boxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in xyxy_boxes]

            for xyxy_box, xywh_box, box_cls, conf in zip(xyxy_boxes, xywh_boxes, cls, confs):
                ann_coco = {
                    "id": len(new_coco_data['annotations']) + 1,
                    "image_id": img_id,
                    "category_id": int(box_cls) + 1,
                    "bbox": [float(x) for x in xywh_box],
                    "xyxy_bbox": [float(x) for x in xyxy_box],
                    "area": 0.0,
                    "score": float(conf),
                    "iscrowd": 0,
                    'segmentation': []
                }
                
                new_coco_data['annotations'].append(ann_coco)

    # save new_coco_data to json
    with open(os.path.join(output_dir, 'od_output.json'), 'w') as f:
        json.dump(new_coco_data, f)

    return new_coco_data


def coco_from_dict(od_json_dict):
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as temp:
        json.dump(od_json_dict, temp)
        coco_json_path = temp.name
    return COCO(coco_json_path)


def run_sam_on_image(sam, sam_trans, img_path, anns, device):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    resize_img = sam_trans.apply_image(image)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam.preprocess(resize_img_tensor[None,:,:,:])
    with torch.no_grad():
        image_embedding = sam.image_encoder(input_image)
    anns_with_masks = []
    for ann in anns:
        bbox = ann['bbox']
        xyxy_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        input_box = torch.tensor([[xyxy_bbox]], device=device)
        input_box = sam_trans.apply_boxes_torch(input_box, (H, W))
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None, boxes=input_box.to(device), masks=None,
            )
            low_res_masks, iou_pred = sam.mask_decoder(
                image_embeddings=image_embedding.squeeze(1).to(device),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        low_res_pred = torch.sigmoid(low_res_masks)
        masks = sam.postprocess_masks(
            low_res_pred,
            input_size=resize_img_tensor.shape[-2:],
            original_size=(H, W),
        )
        score = iou_pred[0].item()
        mask = masks[0].detach().cpu().numpy().squeeze()
        mask = (mask > 0.5).astype(np.uint8)
        polygons = mask_to_single_polygon(mask)
        ann['original_bbox'] = [float(x) for x in bbox]
        ann['od_score'] = ann['score'] if 'score' in ann else 1.0
        ann['score'] = score
        ann['area'] = float(np.sum(mask))
        ann['segmentation'] = polygons
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
            y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))
            ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            ann['bbox'] = [float(x) for x in ann['bbox']]
        anns_with_masks.append(ann)
    return anns_with_masks


def apply_nms_to_annotations(anns_with_masks, constants):
    image_to_anns = {}
    for ann in anns_with_masks:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    filtered_anns = []
    for img_id, img_anns in image_to_anns.items():
        if len(img_anns) <= 1:
            filtered_anns.extend(img_anns)
            continue
        boxes = np.array([ann['bbox'] for ann in img_anns])
        for i in range(len(boxes)):
            if boxes[i, 2] <= 0:
                boxes[i, 2] = 1.0
            if boxes[i, 3] <= 0:
                boxes[i, 3] = 1.0
        xyxy_boxes = np.column_stack([
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 0] + boxes[:, 2],
            boxes[:, 1] + boxes[:, 3]
        ])
        scores = np.array([ann['score'] for ann in img_anns])
        class_ids = np.array([ann['category_id'] - 1 for ann in img_anns])
        nms_processor = NMSProcessor(
            mrna_iou_threshold=constants.SAM_MRNA_IOU_THRESHOLD,
            mrna_size_ratio=constants.SAM_MRNA_SIZE_RATIO,
            mrna_overlap_threshold=constants.SAM_MRNA_OVERLAP_THRESHOLD,
            mrna_containment_threshold=constants.SAM_MRNA_CONTAINMENT_THRESHOLD
        )
        keep_indices = nms_processor._custom_nms(
            boxes=xyxy_boxes,
            scores=scores,
            class_ids=class_ids
        )
        filtered_anns.extend([img_anns[i] for i in keep_indices])
    for idx, ann in enumerate(filtered_anns, start=1):
        ann['id'] = idx
    return filtered_anns


def write_coco_json(output_json_path, coco_dict):
    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f)


def inference_with_sam(model_checkpoint, img_dir, output_dir, od_json_dict):
    """
    Run SAM segmentation on images using bounding boxes from a COCO-format dict.
    Returns: path to output JSON
    """
    coco = coco_from_dict(od_json_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = torch.load(model_checkpoint)
    sam.to(device=device)
    sam.eval()
    sam_trans = ResizeLongestSide(sam.image_encoder.img_size)
    img_ids = coco.getImgIds()
    anns_with_masks = []
    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        anns = coco.loadAnns(coco.getAnnIds(img_id))
        anns_with_masks.extend(run_sam_on_image(sam, sam_trans, img_path, anns, device))
    # filtered_anns = apply_nms_to_annotations(anns_with_masks, constants)
    new_coco_data = copy.deepcopy(od_json_dict)
    new_coco_data['annotations'] = anns_with_masks
    output_json_path = os.path.join(output_dir, 'lnp_mod_output.json')
    write_coco_json(output_json_path, new_coco_data)
    return output_json_path


def run_sam2_on_image(sam, sam_transform, img_path, anns, device, simplify_polygons=True, epsilon=1.0):
    """
    Run SAM2 inference on a single image with the given annotations.
    
    Args:
        sam: The SAM2 model
        sam_transform: SAM2 transform for image preprocessing
        img_path: Path to the input image
        anns: List of annotations to process (with bounding boxes)
        device: Device to run inference on ('cuda' or 'cpu')
        simplify_polygons: Whether to simplify polygons
        epsilon: Epsilon parameter for polygon simplification
        
    Returns:
        List of annotations with added mask data
    """
    if len(anns) == 0:
        print(f"Warning: No annotations found for image: {img_path}")
        return []
        
    # Verify all annotations have bbox
    for i, ann in enumerate(anns):
        if 'bbox' not in ann or len(ann['bbox']) != 4:
            print(f"Warning: Invalid bbox for annotation {i} in {img_path}: {ann.get('bbox')}")
            continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not read image at {img_path}")
        return anns  # Return original annotations without masks
          
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    
    
    
    # Process features with torch.no_grad() to save memory
    try:
        with torch.no_grad():
            # Process image using SAM2 approach
            input_image = sam_transform(image)
            input_image = input_image.unsqueeze(0).to(device)
            
            # Generate embeddings with SAM2 backbone
            backbone_out = sam.forward_image(input_image)
            _, vision_feats, _, _ = sam._prepare_backbone_features(backbone_out)
            
            # Process features in SAM2 format
            if sam.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + sam.no_mem_embed
                
            bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
            ][::-1]
            
            features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    except Exception as e:
        print(f"Error processing image features for {img_path}: {str(e)}")
        return anns  # Return original annotations without masks
    
    # Free up memory
    del backbone_out, vision_feats, input_image
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    img_anns_with_masks = []
    
    for ann_idx, ann in enumerate(tqdm(anns, desc=f'Segmenting Objects')):
        try:
            with torch.no_grad():
                if 'bbox' not in ann or len(ann['bbox']) != 4:
                    print(f"Skipping annotation {ann_idx} with invalid bbox: {ann.get('bbox')}")
                    continue
                    
                bbox = ann['bbox']
                # Store original bbox for reference
                original_bbox = [float(x) for x in bbox]
                

                
                xyxy_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                
                # Check if the bbox is valid
                if (xyxy_bbox[0] >= W or xyxy_bbox[1] >= H or xyxy_bbox[2] <= 0 or xyxy_bbox[3] <= 0):
                    print(f"Warning: Bounding box {xyxy_bbox} is outside image boundaries {W}x{H}")
                    ann_copy = ann.copy()
                    ann_copy['original_bbox'] = original_bbox
                    ann_copy['bbox'] = [float(x) for x in bbox]
                    ann_copy['score'] = 0.0
                    ann_copy['od_score'] = ann['score'] if 'score' in ann else 1.0
                    ann_copy['area'] = 0.0
                    ann_copy['segmentation'] = []
                    img_anns_with_masks.append(ann_copy)
                    continue
                
                # Prepare box input for SAM2
                box = torch.tensor([xyxy_bbox], device=device)
                
                # CUSTOM BOX TRANSFORMATION: Instead of using the transform_boxes method which is failing,
                # we'll manually transform the coordinates to the format SAM2 expects
                
                # 1. Convert box from [x1, y1, x2, y2] to normalized coordinates
                # The target size for SAM2 is 1024x1024
                SAM2_SIZE = getattr(sam_transform, 'resolution', 1024)
                
                # Compute scale factors
                scale_x = SAM2_SIZE / W
                scale_y = SAM2_SIZE / H
                
                # Apply scaling
                x1_norm = xyxy_bbox[0] * scale_x
                y1_norm = xyxy_bbox[1] * scale_y
                x2_norm = xyxy_bbox[2] * scale_x
                y2_norm = xyxy_bbox[3] * scale_y
                
                # Ensure coordinates are within bounds
                x1_norm = max(0.0, min(SAM2_SIZE-1, x1_norm))
                y1_norm = max(0.0, min(SAM2_SIZE-1, y1_norm))
                x2_norm = max(0.0, min(SAM2_SIZE-1, x2_norm))
                y2_norm = max(0.0, min(SAM2_SIZE-1, y2_norm))
                
                # Reshape for SAM2 format which expects [[[x1, y1], [x2, y2]]]
                box_coords = torch.tensor([[[x1_norm, y1_norm], [x2_norm, y2_norm]]], device=device)
                
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
                
                # Generate sparse and dense embeddings
                sparse_embeddings, dense_embeddings = sam.sam_prompt_encoder(
                    points=(box_coords, box_labels),
                    boxes=None,
                    masks=None,
                )
                
                # Get high-res features for current image
                high_res_features = [feat_level[0].unsqueeze(0) for feat_level in features["high_res_feats"]]
                
                # Predict masks using SAM2 decoder
                low_res_masks, iou_predictions, _, _ = sam.sam_mask_decoder(
                    image_embeddings=features["image_embed"][0].unsqueeze(0),
                    image_pe=sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                
                # Process masks to original resolution
                masks = sam_transform.postprocess_masks(
                    low_res_masks, (H, W)
                )
                
                # Move to CPU immediately
                mask = masks[0].detach().cpu().numpy().squeeze()
                mask = (mask > 0.0).astype(np.uint8)
                mask_sum = np.sum(mask)
                score = iou_predictions[0].item()
            
            # Check if mask is valid (contains any foreground pixels)
            if mask_sum == 0:
                print(f"Warning: Empty mask generated for annotation {ann_idx} in {img_path}")
                # For empty masks, keep the original bbox and set empty segmentation
                ann_copy = ann.copy()
                ann_copy['original_bbox'] = original_bbox
                ann_copy['bbox'] = [float(x) for x in bbox]
                ann_copy['score'] = 0.0  # Low score for empty mask
                ann_copy['od_score'] = ann['score'] if 'score' in ann else 1.0
                ann_copy['area'] = 0.0
                ann_copy['segmentation'] = []  # Empty segmentation
                img_anns_with_masks.append(ann_copy)
                continue  # Skip further processing for empty masks
            
            # Process results on CPU to free GPU memory
            # Convert mask to simplified polygon
            polygons = mask_to_single_polygon(mask, simplify=simplify_polygons, epsilon=epsilon)
            
            # Check if polygons were generated properly
            if not polygons:
                print(f"Warning: Failed to generate polygons for annotation {ann_idx} in {img_path}")
                # If polygon generation failed but mask has pixels, try with larger epsilon
                if mask_sum > 0:

                    polygons = mask_to_single_polygon(mask, simplify=True, epsilon=2.0)
                
                # If still no valid polygons, keep original bbox and set empty segmentation
                if not polygons:
                    ann_copy = ann.copy()
                    ann_copy['original_bbox'] = original_bbox
                    ann_copy['bbox'] = [float(x) for x in bbox]
                    ann_copy['score'] = score
                    ann_copy['od_score'] = ann['score'] if 'score' in ann else 1.0
                    ann_copy['area'] = float(mask_sum)
                    ann_copy['segmentation'] = []  # Empty segmentation
                    img_anns_with_masks.append(ann_copy)
                    continue  # Skip further processing for failed polygon generation
            
            # Save the original bbox for debugging
            ann_copy = ann.copy()
            ann_copy['original_bbox'] = original_bbox
            ann_copy['od_score'] = ann['score'] if 'score' in ann else 1.0
            ann_copy['score'] = score
            ann_copy['area'] = float(mask_sum)
            ann_copy['segmentation'] = polygons
            
            # Calculate bbox from mask
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
                y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))
                # COCO format is [x, y, width, height]
                mask_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                ann_copy['bbox'] = mask_bbox
            else:
                # If no mask pixels found, keep original bbox
                print(f"Warning: No mask pixels found for annotation {ann_idx} in {img_path}, keeping original bbox")
                ann_copy['bbox'] = [float(x) for x in bbox]
                
            img_anns_with_masks.append(ann_copy)
            
            # Free up memory after each annotation
            del box, box_coords, box_labels, sparse_embeddings, dense_embeddings
            del low_res_masks, iou_predictions, masks, mask
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error processing annotation {ann_idx} in {img_path}: {str(e)}")
            # Still include the annotation, but without masks
            ann_copy = ann.copy()
            if 'bbox' in ann:
                ann_copy['original_bbox'] = [float(x) for x in ann['bbox']]
                ann_copy['bbox'] = [float(x) for x in ann['bbox']]
                ann_copy['score'] = ann['score'] if 'score' in ann else 0.0
                ann_copy['segmentation'] = []
                img_anns_with_masks.append(ann_copy)
    
    # Free memory after processing the entire image
    del features, feats
    if 'high_res_features' in locals():
        del high_res_features
    
    # Final verification
    if len(img_anns_with_masks) == 0 and len(anns) > 0:
        print(f"Error: No masks generated for any annotations in {img_path}")
    elif len(img_anns_with_masks) < len(anns):
        print(f"Warning: Only {len(img_anns_with_masks)}/{len(anns)} annotations processed for {img_path}")
    
    return img_anns_with_masks


def inference_with_sam2(model_checkpoint, img_dir, output_dir, od_json_dict, 
                       simplify_polygons=True, epsilon=1.0):
    """
    Run inference using SAM 2.1 model to generate masks from bounding boxes.
    
    Args:
        model_checkpoint (str): Path to SAM 2.1 model checkpoint
        img_dir (str): Path to directory containing images
        output_dir (str): Path to directory to save output
        od_json_dict (dict): Object detection results in COCO format
        simplify_polygons (bool): Whether to simplify the polygons (reduce points)
        epsilon (float): Approximation accuracy parameter - higher values create simpler polygons
    
    Returns:
        str: Path to output JSON file with segmentation results
    """
    # Validate input data
    if not od_json_dict or 'images' not in od_json_dict or 'annotations' not in od_json_dict:
        raise ValueError("Invalid object detection results: missing 'images' or 'annotations'")
    
    if len(od_json_dict['images']) == 0:
        raise ValueError("No images found in object detection results")
    
    if len(od_json_dict['annotations']) == 0:
        raise ValueError("No annotations found in object detection results")
    
    # Check if all images have annotations
    image_ids = set(img['id'] for img in od_json_dict['images'])
    ann_image_ids = set(ann['image_id'] for ann in od_json_dict['annotations'])
    missing_images = image_ids - ann_image_ids
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images have no annotations: {missing_images}")
    
    # Create COCO object from input data
    coco = coco_from_dict(od_json_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    sam = torch.load(model_checkpoint)
    if hasattr(sam, 'image_size'):
        image_size = sam.image_size
    else:
        image_size = 1024
    
    sam_transform = SAM2Transforms(
        resolution=image_size,  # SAM2 default resolution
        mask_threshold=0.0,  # Use lowest threshold for EM images
        max_hole_area=0.0,  # No hole filling for precise segmentation
        max_sprinkle_area=0.0  # No sprinkling
    )
    

    
    img_ids = coco.getImgIds()
    if not img_ids:
        raise ValueError("No valid image IDs found in COCO data")
        

    
    # Track overall stats
    total_annotations = 0
    processed_annotations = 0
    anns_with_masks = []

    for i, img_id in enumerate(tqdm(img_ids, desc="Processing images")):

 
        # Clear any existing CUDA memory
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Get annotations for this image before loading model
        img_anns = coco.loadAnns(coco.getAnnIds(img_id))
        total_annotations += len(img_anns)
        
        if not img_anns:
            print(f"Warning: No annotations found for image ID {img_id}")
            continue
            
        # Load model for this image only to manage memory better
        try:
            sam = torch.load(model_checkpoint)
            sam.to(device=device)
            sam.eval()
        except Exception as e:
            print(f"Error loading model for image {img_id}: {str(e)}")
            continue
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        
        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            continue
        
        # Process this image
        try:

            img_anns_with_masks = run_sam2_on_image(
                sam=sam,
                sam_transform=sam_transform,
                img_path=img_path,
                anns=img_anns,
                device=device,
                simplify_polygons=simplify_polygons,
                epsilon=epsilon
            )
            
            processed_annotations += len(img_anns_with_masks)
            
            # Add all masks from current image
            anns_with_masks.extend(img_anns_with_masks)
        except Exception as e:
            print(f"Error processing image {img_info['file_name']}: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Always unload model from GPU after each image
            try:
                sam = sam.to('cpu')
                del sam
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except:
                pass
    

    
    if len(anns_with_masks) == 0:
        # Return original JSON with error message
        error_json_path = os.path.join(output_dir, 'lnp_mod_output_error.json')
        write_coco_json(error_json_path, od_json_dict)
        return error_json_path
    
    # Create output COCO data
    new_coco_data = copy.deepcopy(od_json_dict)
    new_coco_data['annotations'] = anns_with_masks
    

    
    # Write results to file
    output_json_path = os.path.join(output_dir, 'lnp_mod_output.json')
    write_coco_json(output_json_path, new_coco_data)
    
    return output_json_path


