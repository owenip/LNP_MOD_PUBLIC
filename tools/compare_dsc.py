import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import cv2
from collections import defaultdict
import torch

import lnp_mod.utils.utils as utils
import lnp_mod.config.constants as constants
from lnp_mod.utils.load_model import load_model_files
from lnp_mod.core.predict_sam import inference_with_sam, inference_with_sam2


def dice_coefficient(pred_mask, gt_mask):
    """CPU version of dice coefficient"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2. * intersection / (pred.sum() + gt.sum() + 1e-8)


def dice_coefficient_gpu(pred_mask, gt_mask):
    """GPU version of dice coefficient"""
    pred = torch.from_numpy(pred_mask.astype(np.uint8)).cuda().bool()
    gt = torch.from_numpy(gt_mask.astype(np.uint8)).cuda().bool()
    
    intersection = torch.logical_and(pred, gt).sum().float()
    return (2. * intersection / (pred.sum() + gt.sum() + 1e-8)).cpu().item()


def batch_dice_coefficient_gpu(pred_masks, gt_masks, batch_size=32):
    """Calculate DSC for multiple prediction-GT mask pairs on GPU in batches"""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    max_dsc_per_pred = np.zeros(n_pred)
    
    # Process in batches to avoid OOM
    for i in range(0, n_pred, batch_size):
        pred_batch = torch.from_numpy(np.stack(pred_masks[i:i+batch_size])).cuda().bool()
        best_dsc_batch = torch.zeros(len(pred_batch), device='cuda')
        
        for j in range(0, n_gt, batch_size):
            gt_batch = torch.from_numpy(np.stack(gt_masks[j:j+batch_size])).cuda().bool()
            
            # Calculate pairwise DSC scores
            for p_idx, pred in enumerate(pred_batch):
                for g_idx, gt in enumerate(gt_batch):
                    intersection = torch.logical_and(pred, gt).sum().float()
                    dsc = 2. * intersection / (pred.sum() + gt.sum() + 1e-8)
                    best_dsc_batch[p_idx] = torch.max(best_dsc_batch[p_idx], dsc)
        
        max_dsc_per_pred[i:i+len(pred_batch)] = best_dsc_batch.cpu().numpy()
    
    return max_dsc_per_pred


def get_gt_mask(coco, img_info, img_shape):
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for ann in anns:
        m = coco.annToMask(ann)
        mask = np.maximum(mask, m)
    return mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='Input directory path', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory path', default=None)
    parser.add_argument('-c', '--coco_json_path', help='Path to COCO annotation JSON', required=True)
    parser.add_argument('--seg_model', help='Segmentation model to use', required=True)
    parser.add_argument('--output_csv', help='Path to save DSC scores as CSV', default=None)
    parser.add_argument('--use_gpu', help='Use GPU for DSC calculation', action='store_true', default=True)
    parser.add_argument('--batch_size', help='Batch size for GPU operations', type=int, default=32)
    return parser.parse_args()


def prepare_output_dir(input_dir, output_dir):
    input_dir = os.path.join(input_dir, '')
    if output_dir is None:
        output_dir = utils.create_output_folder(input_dir)
    else:
        output_dir = os.path.join(output_dir, '')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    return input_dir, output_dir


def load_coco_annotations(coco_json_path):
    if not os.path.exists(coco_json_path):
        print(f"COCO annotation JSON file not found: {coco_json_path}")
        exit(1)
    coco = COCO(coco_json_path)
    return coco


def load_segmentation_model(seg_model_name):
    if seg_model_name not in constants.SEG_MODEL_FILES:
        print(f"Segmentation model {seg_model_name} not found. Available: {list(constants.SEG_MODEL_FILES.keys())}")
        exit(1)
    seg_model = load_model_files(
        os.path.join(os.getcwd(), 'models', ''),
        constants.SEG_MODEL_FILES[seg_model_name]
    )
    return seg_model


def run_inference(seg_model_name, seg_model, input_dir, output_dir, coco):
    print("Running SAM on images...")
    if seg_model_name.startswith('sam2'):
        return inference_with_sam2(
            model_checkpoint=seg_model,
            img_dir=input_dir,
            output_dir=output_dir,
            od_json_dict=coco.dataset,
            simplify_polygons=True,
            epsilon=1.0
        )
    elif seg_model_name.startswith('sam') or seg_model_name.startswith('segmentation_model'):
        return inference_with_sam(
            model_checkpoint=seg_model,
            img_dir=input_dir,
            output_dir=output_dir,
            od_json_dict=coco.dataset,
        )
    else:
        raise ValueError(f"Invalid segmentation model: {seg_model_name}")


def evaluate_annotations(pred_coco, coco, input_dir, use_gpu, categories, batch_size=32):
    results = []
    all_ann_dscs = []
    category_dscs = defaultdict(list)
    pred_ann_ids = pred_coco.getAnnIds()
    pred_anns = pred_coco.loadAnns(pred_ann_ids)
    img_to_anns = defaultdict(list)
    for pred_ann in pred_anns:
        img_to_anns[pred_ann['image_id']].append(pred_ann)
    all_pred_anns = [ann for anns in img_to_anns.values() for ann in anns]
    image_cache = {}
    gt_masks_cache = {}  # Cache by image_id and annotation_id
    
    # Set to track processed annotation IDs for cache management
    processed_ann_ids = set()
    
    def get_image(img_path):
        if img_path not in image_cache:
            image_cache[img_path] = utils.imread(img_path)
        return image_cache[img_path]
    
    # First, build a dictionary of GT annotations by ID
    print("Building GT annotation lookup...")
    gt_anns_by_id = {}
    for ann_id in tqdm(coco.getAnnIds(), desc="Loading GT annotations"):
        ann = coco.loadAnns(ann_id)[0]
        gt_anns_by_id[ann_id] = ann
    
    def get_gt_mask(ann_id):
        if ann_id not in gt_anns_by_id:
            return None
        
        if ann_id not in gt_masks_cache:
            try:
                gt_masks_cache[ann_id] = coco.annToMask(gt_anns_by_id[ann_id])
            except Exception as e:
                print(f"Warning: Invalid GT annotation {ann_id}: {e}")
                gt_masks_cache[ann_id] = None
        
        return gt_masks_cache[ann_id]
    
    def clear_caches(counter):
        """Clear caches periodically to manage memory usage"""
        if counter % 100 == 0 and counter > 0:
            # Report memory usage if psutil is available
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                print(f"\nMemory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
            except ImportError:
                pass
            
            # Clear image cache completely
            print(f"\nClearing image cache at annotation {counter}/{len(all_pred_anns)}")
            image_cache.clear()
            
            # Clear GT masks for annotations we've already processed
            masks_to_remove = [ann_id for ann_id in gt_masks_cache if ann_id in processed_ann_ids]
            for ann_id in masks_to_remove:
                del gt_masks_cache[ann_id]
            
            print(f"Cleared {len(masks_to_remove)} GT masks from cache")
    
    print(f"Processing {len(all_pred_anns)} predicted annotations...")
    for counter, pred_ann in enumerate(tqdm(all_pred_anns, desc="Evaluating annotations")):
        # Periodically clear caches to manage memory
        clear_caches(counter)
        
        ann_id = pred_ann['id']
        img_id = pred_ann['image_id']
        img_info = pred_coco.loadImgs(img_id)[0]
        img_path = os.path.join(input_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        img = get_image(img_path)
        
        # Find the corresponding GT mask with matching annotation ID
        gt_mask = get_gt_mask(ann_id)
        
        if gt_mask is None:
            print(f"Warning: No matching GT mask for annotation ID {ann_id}")
            # Set DSC to 0 when there's no matching GT mask
            best_dsc = 0
            category_id = pred_ann['category_id']
            category_dscs[category_id].append(best_dsc)
            all_ann_dscs.append(best_dsc)
            results.append({
                'image': img_info['file_name'],
                'annotation_id': ann_id,
                'category_id': category_id,
                'category': categories.get(category_id, f"Unknown-{category_id}"),
                'dsc': best_dsc,
                'match_type': 'no_gt_match'
            })
            # Mark this annotation as processed
            processed_ann_ids.add(ann_id)
            continue
        
        # Create predicted mask
        if not pred_ann.get('segmentation') or len(pred_ann['segmentation']) == 0:
            pred_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask_type = 'empty'
        else:
            try:
                pred_mask = pred_coco.annToMask(pred_ann)
                mask_type = 'valid'
            except Exception as e:
                print(f"Warning: Invalid pred annotation {ann_id}: {e}")
                pred_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask_type = 'invalid'
        
        # Calculate DSC only against the matching GT mask
        if use_gpu:
            try:
                dsc = dice_coefficient_gpu(pred_mask, gt_mask)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in GPU dice calculation: {e}")
                dsc = dice_coefficient(pred_mask, gt_mask)
        else:
            dsc = dice_coefficient(pred_mask, gt_mask)
        
        # Store result
        category_id = pred_ann['category_id']
        category_dscs[category_id].append(dsc)
        all_ann_dscs.append(dsc)
        results.append({
            'image': img_info['file_name'],
            'annotation_id': ann_id,
            'category_id': category_id,
            'category': categories.get(category_id, f"Unknown-{category_id}"),
            'dsc': dsc,
            'match_type': 'exact_id_match',
            'mask_type': mask_type
        })
        
        # Mark this annotation as processed
        processed_ann_ids.add(ann_id)
    
    # Final memory report
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"\nFinal memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    except ImportError:
        pass
    
    return results, all_ann_dscs, category_dscs


def print_summary(all_ann_dscs, category_dscs, categories):
    all_ann_dscs = np.array(all_ann_dscs)
    print("\n=== OVERALL DSC STATISTICS ===")
    print(f"Mean DSC across all annotations: {all_ann_dscs.mean():.4f}")
    print(f"Std DSC across all annotations: {all_ann_dscs.std():.4f}")
    print(f"Median DSC across all annotations: {np.median(all_ann_dscs):.4f}")
    print(f"Min DSC: {all_ann_dscs.min():.4f}")
    print(f"Max DSC: {all_ann_dscs.max():.4f}")
    print("\n=== PER-CATEGORY DSC STATISTICS ===")
    print(f"{'Category':<20} {'Count':<8} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8}")
    print("-" * 65)
    for category_id, dscs in sorted(category_dscs.items()):
        category_name = categories.get(category_id, f"Unknown-{category_id}")
        dscs_array = np.array(dscs)
        print(f"{category_name:<20} {len(dscs):<8} {dscs_array.mean():.4f}  {np.median(dscs_array):.4f}  {dscs_array.min():.4f}  {dscs_array.max():.4f}")


def main():
    args = parse_args()
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and not use_gpu:
        print("Warning: GPU requested but not available. Using CPU instead.")
    elif use_gpu:
        print(f"Using GPU for DSC calculation: {torch.cuda.get_device_name(0)}")
    input_dir, output_dir = prepare_output_dir(args.input_dir, args.output_dir)
    coco = load_coco_annotations(args.coco_json_path)
    seg_model = load_segmentation_model(args.seg_model)
    pred_json_path = run_inference(args.seg_model, seg_model, input_dir, output_dir, coco)
    pred_coco = COCO(pred_json_path)
    overlay_dir = os.path.join(output_dir, 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)
    categories = {cat['id']: cat['name'] for cat in pred_coco.dataset['categories']}
    if use_gpu:
        torch.cuda.empty_cache()
    results, all_ann_dscs, category_dscs = evaluate_annotations(pred_coco, coco, input_dir, use_gpu, categories, batch_size=args.batch_size)
    output_csv = args.output_csv or os.path.join(output_dir, 'dsc_scores.csv')
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"DSC scores saved to {output_csv}")
    print_summary(all_ann_dscs, category_dscs, categories)


def get_pred_mask(pred_coco, img_info, img_shape):
    ann_ids = pred_coco.getAnnIds(imgIds=img_info['id'])
    anns = pred_coco.loadAnns(ann_ids)
    pred_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for ann in anns:
        # If segmentation is empty, skip adding to mask (mask remains zero)
        if not ann.get('segmentation') or len(ann['segmentation']) == 0:
            continue
        try:
            m = pred_coco.annToMask(ann)
            pred_mask = np.maximum(pred_mask, m)
        except Exception as e:
            print(f"Warning: Invalid annotation for image {img_info['file_name']}: {e}")
            # Optionally continue, mask remains zero
    return pred_mask


if __name__ == "__main__":
    main()
