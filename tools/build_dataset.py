import shutil
import numpy as np
import yaml
import os
import json
import argparse
from lnp_mod.config import constants
import shlex
import cv2
from pycocotools.coco import COCO
import tqdm
import albumentations as A
import logging
import multiprocessing as mp
from multiprocessing import Manager, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union
import traceback
from multiprocessing.managers import ValueProxy, AcquirerProxy

# Constants
TRAIN_SPLIT_RATIO = 0.8
MIN_BBOX_AREA = 10
MIN_BBOX_VISIBILITY = 0.1
PATCH_SIZE = 2048
PATCH_OVERLAP = 1024
NUM_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one CPU free

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Take args from cml --dataset_yaml_path --output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_yaml_path", help="Dataset yaml file path")
    parser.add_argument("--output_dir", help="Output directory path")

    # Optional arguments
    # whether keep as original or patchify the dataset, default is False
    parser.add_argument("--patchify", help="Patchify the dataset", action="store_true")
    # class ids to merge, default is all id in  constants.SEGMENTATION_CATEGORIES.items()
    parser.add_argument("--cat_ids", help="Category_ids to merge", nargs='*', type=int)
    # Crop all objects from the image, default is False
    parser.add_argument("--crop_all_obj", help="Crop the objects from the image", action="store_true")
    # Whether split the dataset into train and val, default is False
    parser.add_argument("--split", help="Split the dataset into train and val", action="store_true")
    # Whether save the bbox information only, default is False
    parser.add_argument("--bbox_only", help="Only save the bbox information", action="store_true")

    args = parser.parse_args()
    dataset_yaml_path = args.dataset_yaml_path
    output_dir = args.output_dir

    cat_ids = args.cat_ids
    if cat_ids is None:
        cat_ids = list(constants.SEGMENTATION_CATEGORIES.keys())

    is_patchify = args.patchify
    is_crop_obj = args.crop_all_obj
    is_split = args.split
    is_bbox_only = args.bbox_only

    if is_patchify and is_crop_obj:
        raise ValueError('Cannot crop all objects and patchify the dataset at the same time')

    process(dataset_yaml_path, output_dir, is_patchify, cat_ids, is_crop_obj, is_split, is_bbox_only)


def process(dataset_yaml_path, output_dir, is_patchify=False, cat_ids=None, is_crop_all_obj=False, is_split=False, is_bbox_only=False):
    yaml_data = read_yaml_file(dataset_yaml_path)
    images_sets = get_images_sets(yaml_data)
    check_images_sets(images_sets)

    output_dir, train_dir, val_dir = create_output_folders(output_dir, is_split)
    if is_crop_all_obj:
        print('Feature not implemented')
    elif is_patchify:
        if is_split:
            patchify_and_merge_and_split_dataset(images_sets, cat_ids, train_dir, val_dir, is_bbox_only=is_bbox_only)
        else:
            patchify_and_merge_datasets(images_sets, cat_ids, output_dir, is_bbox_only=is_bbox_only)
    else:
        if is_split:
            merge_and_split_datasets(images_sets, cat_ids, train_dir, val_dir, is_bbox_only=is_bbox_only)
        else:
            merge_datasets(images_sets, cat_ids, output_dir, is_bbox_only=is_bbox_only)


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data


def get_images_sets(yaml_data):
    return yaml_data['images_sets']

def check_images_sets(images_sets):
    for images_set in images_sets:
        images_set_name = images_set['name']
        images_set_images = images_set['images_dir']
        images_set_coco = images_set['coco_file']

        images_set_images = os.path.join(images_set_images, '')
        if not os.path.exists(images_set_images):
            raise FileNotFoundError(f'Error: Reading {images_set_name}\n{images_set_images} does not exist')

        if not os.path.exists(images_set_coco):
            raise FileNotFoundError(f'Error: Reading {images_set_name}\n{images_set_coco} does not exist')

    return True


def create_output_folders(tar_dir, is_split=False):
    tar_dir = os.path.join(tar_dir, '')
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    
    if is_split is False:
        img_dir = os.path.join(tar_dir, 'images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # delete all images
        for file in os.listdir(img_dir):
            if os.path.isfile(os.path.join(img_dir, file)):
                os.remove(os.path.join(img_dir, file))

    if is_split:
        train_dir = os.path.join(tar_dir, 'train')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        val_dir = os.path.join(tar_dir, 'val')
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        train_img_dir = os.path.join(train_dir, 'images')
        if not os.path.exists(train_img_dir):
            os.makedirs(train_img_dir)
        else:
            for file in os.listdir(train_img_dir):
                os.remove(os.path.join(train_img_dir, file))
        
        val_img_dir = os.path.join(val_dir, 'images')
        if not os.path.exists(val_img_dir):
            os.makedirs(val_img_dir)
        else:
            for file in os.listdir(val_img_dir):
                os.remove(os.path.join(val_img_dir, file))
    else:
        train_dir = tar_dir
        val_dir = tar_dir    
    

    return tar_dir, train_dir, val_dir


def merge_with_crop_all_objects(dataset, output_images_dir, new_coco_data, class_ids):
    dataset_name = dataset['name']
    dataset_images = dataset['images_dir']
    dataset_coco = COCO(dataset['coco_file'])

    dataset_images = os.path.join(dataset_images, '')

    # iterate through the images from dataset_coco and crop all the objects
    for img in dataset_coco.imgs.values():
        img_id = img['id']
        image_path = os.path.join(dataset_images, img['file_name'])
        image_name = img['file_name'].split('.')[0]

        # Read the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read the coco file
        anns = dataset_coco.loadAnns(dataset_coco.getAnnIds(imgIds=img_id, catIds=class_ids))
        for annotation in anns:
            bbox = annotation['bbox']
            x, y, w, h = bbox

            # Crop the image at the bbox
            cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
            # Check if the cropped image is empty
            if cropped_img.size == 0:
                continue
            
            # Save the cropped image
            img_id = len(new_coco_data['images']) + 1
            img_filename = f'{img_id + 1}.png'
            cropped_img_path = os.path.join(output_images_dir, img_filename)

            try:
                cv2.imwrite(cropped_img_path, cropped_img)
            except Exception as e:
                print(f'Error: {e}')
                print(f'Error: Saving {cropped_img_path}')
                print(f'image: {image_path}')
                print('annotation:', annotation)
                continue

            image_coco = {
                "id": len(new_coco_data['images']) + 1,
                "file_name": img_filename,
                "dataset": dataset_name,
                "width": w,
                "height": h
            }
            new_coco_data['images'].append(image_coco)
            
            ann_coco = {
                "id": len(new_coco_data['annotations']) + 1,
                "image_id": image_coco['id'],
                "category_id": annotation['category_id'] + 1,
                "bbox": [0, 0, w, h],
                "area": w * h,
                "segmentation": [[float(coord) - bbox[i%2] for i, coord in enumerate(annotation['segmentation'][0])]],
                "iscrowd": 0
            }

            new_coco_data['annotations'].append(ann_coco)


def merge_datasets(images_sets, category_ids, output_dir, is_bbox_only=False):
    new_coco_data = get_coco_data_template(category_ids)

    pbar = tqdm.tqdm(images_sets, position=0)
    pbar.set_description(f'Merging dataset')
    for images_set in pbar:
        pbar.set_postfix_str(f'{images_set["name"]}')
        merge_dataset(images_set, output_dir, new_coco_data, category_ids, image_ids=None, is_bbox_only=is_bbox_only)
    
    save_coco_file(new_coco_data, output_dir, 'merge_coco.json')


def merge_and_split_datasets(images_sets, category_ids, train_dir, val_dir, is_bbox_only=False):
    if not images_sets:
        raise ValueError("No image sets provided")
    
    train_split_ratio = 0.8
    train_coco_data = get_coco_data_template(category_ids)
    val_coco_data = get_coco_data_template(category_ids)
    
    pbar = tqdm.tqdm(images_sets, position=0)
    pbar.set_description(f'Merging and Splitting dataset')
    for images_set in pbar:
        pbar.set_postfix_str(f'{images_set["name"]}')
        images_set_coco = COCO(images_set['coco_file'])
        image_ids = images_set_coco.getImgIds()
        split_index = int(len(image_ids) * train_split_ratio)
        train_image_ids = image_ids[:split_index]
        val_image_ids = image_ids[split_index:]

        merge_dataset(images_set, train_dir, train_coco_data, category_ids, train_image_ids, is_bbox_only=is_bbox_only)
        merge_dataset(images_set, val_dir, val_coco_data, category_ids, val_image_ids, is_bbox_only=is_bbox_only)

    save_coco_file(train_coco_data, train_dir, 'train_coco.json')
    save_coco_file(val_coco_data, val_dir, 'val_coco.json')


def merge_dataset(image_set_data, output_dir, new_coco_data, category_ids, image_ids=None, is_bbox_only=False):
    images_set_coco = COCO(image_set_data['coco_file'])
    images_dir = os.path.join(image_set_data['images_dir'], '')
    images_set_name = image_set_data['name']
    img_ids = image_ids if image_ids is not None else images_set_coco.getImgIds()
    imgs = images_set_coco.loadImgs(img_ids)

    # Create mapping of category id from image_set_coco to categories in new_coco_data base on common name
    set_len = len(images_set_coco.loadCats(images_set_coco.getCatIds()))
    new_cat_len = len(category_ids)
    reorder_cats = len(images_set_coco.loadCats(images_set_coco.getCatIds())) != len(category_ids)
    if reorder_cats:
        cat_id_map = get_category_id_mapping(images_set_coco.loadCats(category_ids), category_ids, new_coco_data['categories'])

    pb = tqdm.tqdm(imgs, position=1)
    pb.set_description(f'Merging {images_set_name} dataset')
    for img in pb:
        pb.set_postfix_str(f'{img["file_name"]}')
        image_path = os.path.join(images_dir, img['file_name'])
        image = cv2.imread(image_path)
        image_output_filepath = os.path.join(output_dir, 'images', img['file_name'])
        cv2.imwrite(image_output_filepath, image)

        old_img_id = img['id']
        img['id'] = len(new_coco_data['images']) + 1
        img['dataset'] = images_set_name
        new_coco_data['images'].append(img)

        ann_ids = images_set_coco.getAnnIds(imgIds=old_img_id)
        anns = images_set_coco.loadAnns(ann_ids)
        for ann in anns:
            if ann['category_id'] not in category_ids:
                continue

            new_ann = {
                "id": len(new_coco_data['annotations']) + 1,
                "image_id": img['id'],
                "category_id": cat_id_map[ann['category_id']] if reorder_cats else ann['category_id'],
                "bbox": ann['bbox'],
                "area": float(ann['bbox'][2] * ann['bbox'][3]),
                "iscrowd": 0
            }
            
            if not is_bbox_only:
                new_ann["segmentation"] = ann['segmentation']

            new_coco_data['annotations'].append(new_ann)


def patchify_and_merge_datasets(images_sets, category_ids, output_dir, is_bbox_only=False):
    new_coco_data = get_coco_data_template(category_ids)

    pbar = tqdm.tqdm(images_sets, position=0)
    pbar.set_description(f'Patchify and Merging dataset')
    for images_set in pbar:
        pbar.set_postfix_str(f'{images_set["name"]}')
        patchify_and_merge_dataset(images_set, output_dir, new_coco_data, category_ids, image_ids=None, is_bbox_only=is_bbox_only)

    save_coco_file(new_coco_data, output_dir, 'merge_patchify_coco.json')


def patchify_and_merge_and_split_dataset(images_set_data, category_ids, train_dir, val_dir, update_edge_ann=True, is_bbox_only=False):
    train_split_ratio = 0.8
    train_coco_data = get_coco_data_template(category_ids)
    val_coco_data = get_coco_data_template(category_ids)

    pbar = tqdm.tqdm(images_set_data, position=1)
    pbar.set_description(f'Patchify and Merging dataset')
    for images_set in pbar:
        pbar.set_postfix_str(f'{images_set["name"]}')
        images_set_coco = COCO(images_set['coco_file'])
        image_ids = images_set_coco.getImgIds()
        split_index = int(len(image_ids) * train_split_ratio)
        train_image_ids = image_ids[:split_index]
        val_image_ids = image_ids[split_index:]

        patchify_and_merge_dataset(images_set, train_dir, train_coco_data, category_ids, train_image_ids, update_edge_ann, is_bbox_only)
        patchify_and_merge_dataset(images_set, val_dir, val_coco_data, category_ids, val_image_ids, update_edge_ann, is_bbox_only)
        pbar.update(1)

    save_coco_file(train_coco_data, train_dir, 'patchify_train_coco.json')
    save_coco_file(val_coco_data, val_dir, 'patchify_val_coco.json')


def patchify_and_merge_dataset(images_set_data, output_dir, new_coco_data, category_ids, image_ids=None, update_edge_ann=True, is_bbox_only=False):
    output_img_dir = os.path.join(output_dir, 'images')
    patchify_images_and_annotations(
        images_set_data,
        output_img_dir,
        new_coco_data,
        category_ids,
        image_ids,
        update_edge_ann,
        is_bbox_only
    )


def patchify_images_and_annotations(images_set_data, output_img_dir, new_coco_data, category_ids, img_ids=None, update_edge_ann=True, is_bbox_only=False):
    try:
        crop_functions = get_crop_functions()
        images_set_name = images_set_data['name']
        images_dir = os.path.join(images_set_data['images_dir'], '')
        images_set_coco = COCO(images_set_data['coco_file'])
        
        # Get image IDs and load images
        img_ids = img_ids if img_ids is not None else images_set_coco.getImgIds()
        original_imgs = images_set_coco.loadImgs(img_ids)

        # Create shared counters using Manager
        manager = mp.Manager()
        try:
            # Initialize counters based on existing data
            max_image_id = max((img['id'] for img in new_coco_data['images']), default=0)
            max_annotation_id = max((ann['id'] for ann in new_coco_data['annotations']), default=0)
            
            image_id_counter = manager.Value('i', max_image_id)
            annotation_id_counter = manager.Value('i', max_annotation_id)
            lock = manager.Lock()
            
            # Process images in parallel
            new_images = []
            new_annotations = []
            
            # Create batches of images to process to manage memory better
            BATCH_SIZE = 10
            total_batches = (len(original_imgs) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for i in range(0, len(original_imgs), BATCH_SIZE):
                batch_imgs = original_imgs[i:i + BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{total_batches}")
                
                with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = []
                    for img in batch_imgs:
                        try:
                            img_anns = images_set_coco.loadAnns(images_set_coco.getAnnIds(imgIds=img['id'], catIds=category_ids))
                            # Add masks to annotations to avoid COCO API calls in child processes
                            for ann in img_anns:
                                try:
                                    ann['mask'] = images_set_coco.annToMask(ann)
                                except Exception as e:
                                    logger.error(f"Error creating mask for annotation {ann['id']}: {str(e)}")
                                    continue
                            
                            img_data = {
                                'id': img['id'],
                                'file_name': img['file_name'],
                                'image_path': os.path.join(images_dir, img['file_name']),
                                'dataset': images_set_name,
                                'annotations': img_anns
                            }
                            
                            futures.append(executor.submit(
                                process_single_image,
                                img_data,
                                output_img_dir,
                                crop_functions,
                                category_ids,
                                update_edge_ann,
                                is_bbox_only,
                                image_id_counter,
                                annotation_id_counter,
                                lock
                            ))
                        except Exception as e:
                            logger.error(f"Error preparing image {img['file_name']} for processing: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            continue
                    
                    with tqdm.tqdm(total=len(futures), desc=f"Processing batch {i//BATCH_SIZE + 1}/{total_batches}") as pbar:
                        for future in as_completed(futures):
                            try:
                                imgs, anns = future.result()
                                new_images.extend(imgs)
                                new_annotations.extend(anns)
                                pbar.update(1)
                            except Exception as e:
                                logger.error(f"Error in parallel processing: {str(e)}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                pbar.update(1)
                                continue

            # Update COCO data
            new_coco_data['images'].extend(new_images)
            new_coco_data['annotations'].extend(new_annotations)

            # Log summary
            logger.info(f"Processed {len(original_imgs)} images")
            logger.info(f"Created {len(new_images)} new images")
            logger.info(f"Created {len(new_annotations)} new annotations")

        finally:
            # Clean up manager
            manager.shutdown()

    except Exception as e:
        logger.error(f"Error in patchify_images_and_annotations: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def process_single_image(
    image_data: Dict[str, Any],
    output_img_dir: str,
    crop_functions: List[A.Compose],
    category_ids: List[int],
    update_edge_ann: bool,
    is_bbox_only: bool,
    image_id_counter: ValueProxy,
    annotation_id_counter: ValueProxy,
    lock: AcquirerProxy
) -> Tuple[List[Dict], List[Dict]]:
    """Process a single image and its annotations
    
    Args:
        image_data: Dictionary containing image information and annotations
        output_img_dir: Directory to save processed images
        crop_functions: List of albumentations transforms for cropping
        category_ids: List of category IDs to process
        update_edge_ann: Whether to update edge annotations
        is_bbox_only: Whether to process bbox annotations only
        image_id_counter: Shared counter for image IDs
        annotation_id_counter: Shared counter for annotation IDs
        lock: Lock for thread-safe operations
    
    Returns:
        Tuple of (new_images, new_annotations) lists containing the processed data
    """
    try:
        new_images = []
        new_annotations = []
        image_path = image_data['image_path']
        
        # Get edge category ID if needed
        edge_cat_id = None
        if update_edge_ann:
            edge_cat_id = constants.SEG_TO_COCO_CATEGORIES_MAPPING[constants.SEG_NOT_FULLY_VISIBLE_LNP]

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image {image_path}")
            return [], []
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get annotations
        anns = image_data['annotations']
        if not anns:
            logger.warning(f"No annotations found for image {image_path}")
            return [], []

        # Prepare data for augmentation
        bboxes = []
        masks = []
        ann_cat_ids = []
        original_anns = []

        for ann in anns:
            try:
                mask = ann['mask']
                if is_bbox_only or mask.sum() > 0:
                    masks.append(mask)
                    bboxes.append(ann['bbox'])
                    ann_cat_ids.append(ann['category_id'])
                    original_anns.append(ann)
            except Exception as e:
                logger.warning(f"Error processing annotation {ann.get('id', 'unknown')}: {str(e)}")
                continue

        if not bboxes:
            logger.warning(f"No valid bboxes found for image {image_path}")
            return [], []

        # Validate image dimensions
        if img.shape[0] < PATCH_SIZE or img.shape[1] < PATCH_SIZE:
            logger.warning(f"Image {image_path} is smaller than patch size ({img.shape[0]}x{img.shape[1]} < {PATCH_SIZE}x{PATCH_SIZE})")
            return [], []

        # Add validation for bboxes
        valid_indices = []
        for i, bbox in enumerate(bboxes):
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            valid_indices.append(i)

        if not valid_indices:
            logger.warning(f"No valid bboxes after dimension check in {image_path}")
            return [], []

        # Filter valid data
        bboxes = [bboxes[i] for i in valid_indices]
        masks = [masks[i] for i in valid_indices]
        ann_cat_ids = [ann_cat_ids[i] for i in valid_indices]
        original_anns = [original_anns[i] for i in valid_indices]

        # Apply each crop transformation
        for crop_idx, crop_function in enumerate(crop_functions):
            try:
                # Apply transformation
                transformed = crop_function(
                    image=img,
                    masks=masks,
                    bboxes=bboxes,
                    category_ids=ann_cat_ids
                )
                
                if not transformed['bboxes']:
                    continue

                # Count valid annotations before allocating IDs
                valid_annotations = [(bbox, mask, cat_id) for bbox, mask, cat_id in zip(
                    transformed['bboxes'],
                    transformed['masks'],
                    transformed.get('category_ids', ann_cat_ids[:len(transformed['bboxes'])])
                ) if is_bbox_only or mask.any()]

                if not valid_annotations:
                    continue

                # Pre-allocate all needed IDs atomically in a single lock operation
                with lock:
                    image_id_counter.value += 1
                    new_img_id = image_id_counter.value
                    
                    # Pre-allocate annotation IDs
                    num_valid_annotations = len(valid_annotations)
                    first_ann_id = annotation_id_counter.value + 1
                    annotation_id_counter.value += num_valid_annotations
                    annotation_ids = list(range(first_ann_id, first_ann_id + num_valid_annotations))
                    
                    # Generate image name inside the lock to ensure uniqueness
                    dataset_prefix = image_data['dataset'].replace(' ', '_').replace('-', '_')
                    base_name = os.path.splitext(image_data["file_name"])[0]
                    ext = os.path.splitext(image_data["file_name"])[1]
                    img_name = f'{new_img_id}_{dataset_prefix}_{base_name}_patch{crop_idx}{ext}'
                    cropped_img_path = os.path.join(output_img_dir, img_name)

                # Create image data
                new_image = {
                    "id": new_img_id,
                    "file_name": img_name,
                    "dataset": image_data['dataset'],
                    "width": transformed['image'].shape[1],
                    "height": transformed['image'].shape[0]
                }
                
                # Write image file
                try:
                    cv2.imwrite(cropped_img_path, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                except Exception as e:
                    logger.error(f"Error saving image {cropped_img_path}: {str(e)}")
                    continue
                
                new_images.append(new_image)

                # Process annotations with pre-allocated IDs
                batch_annotations = []
                for ann_idx, (bbox, mask, cat_id) in enumerate(valid_annotations):
                    try:
                        bbox = [float(coord) for coord in bbox]
                        
                        if update_edge_ann and edge_cat_id in category_ids:
                            cat_id = get_cat_id_if_on_edge(
                                cat_id,
                                bbox,
                                edge_cat_id,
                                transformed['image'].shape[1],
                                transformed['image'].shape[0]
                            )

                        new_ann = {
                            "id": annotation_ids[ann_idx],
                            "dataset": image_data['dataset'],
                            "image_id": new_img_id,
                            "category_id": cat_id,
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),
                            "iscrowd": 0
                        }

                        if not is_bbox_only:
                            segments = masks2segments(np.array([mask]))
                            if segments and segments[0].size > 0:
                                new_ann["segmentation"] = [segments[0].flatten().tolist()]
                            else:
                                continue

                        batch_annotations.append(new_ann)
                    except Exception as e:
                        logger.warning(f"Error processing annotation {ann_idx} in crop {crop_idx}: {str(e)}")
                        continue
                
                new_annotations.extend(batch_annotations)

            except Exception as e:
                logger.warning(f"Error in crop {crop_idx} for {image_path}: {str(e)}")
                continue

        return new_images, new_annotations

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], []

def get_crop_functions():
    bbox_params = A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_area=MIN_BBOX_AREA,
        min_visibility=MIN_BBOX_VISIBILITY
    )
    
    transforms = [
        ('upper_left', 0, 0),
        ('upper_center', PATCH_OVERLAP, 0),
        ('upper_right', PATCH_SIZE, 0),
        ('center_left', 0, PATCH_OVERLAP),
        ('center_center', PATCH_OVERLAP, PATCH_OVERLAP),
        ('center_right', PATCH_SIZE, PATCH_OVERLAP),
        ('lower_left', 0, PATCH_SIZE),
        ('lower_center', PATCH_OVERLAP, PATCH_SIZE),
        ('lower_right', PATCH_SIZE, PATCH_SIZE),
    ]
    
    crop_functions = []
    for _, x_min, y_min in transforms:
        transform = A.Compose([
            A.Crop(
                x_min=x_min,
                y_min=y_min,
                x_max=x_min + PATCH_SIZE,
                y_max=y_min + PATCH_SIZE
            ),
        ], bbox_params=bbox_params)
        crop_functions.append(transform)
    
    return crop_functions


def get_coco_data_template(category_ids):
    return {
        "info": {
            "description": "Merged COCO Dataset",
        },
        "categories": get_categories(category_ids),
        "images" : [],
        "annotations" : []
    }


def get_categories(category_ids):
    """Create categories list ensuring proper ordering and mapping.
    
    Args:
        category_ids: List of category IDs to include
        
    Returns:
        List of category dictionaries with proper IDs and names
    """
    # First create a mapping of all possible categories
    all_categories = {
        cat_id: {"id": cat_id, "name": constants.SEGMENTATION_CATEGORIES[cat_id]}
        for cat_id in constants.SEGMENTATION_CATEGORIES.keys()
    }
    
    # If we're using all categories, return them in original order
    if set(category_ids) == set(constants.SEGMENTATION_CATEGORIES.keys()):
        return [all_categories[cat_id] for cat_id in category_ids]
        
    # If we're using a subset, create new sequential IDs while preserving names
    categories = []
    for idx, category_id in enumerate(category_ids, 1):
        if category_id not in all_categories:
            logger.warning(f"Category ID {category_id} not found in SEGMENTATION_CATEGORIES")
            continue
        categories.append({
            "id": idx,
            "name": all_categories[category_id]["name"]
        })
    
    return categories

def get_category_id_mapping(old_categories, selected_category_ids, new_categories):
    mapping = {}
    for category in old_categories:
        if category['id'] not in selected_category_ids:
            continue

        for new_cat in new_categories:
            if category['name'] == new_cat['name']:
                mapping[category['id']] = new_cat['id']
                break

    if len(mapping) != len(selected_category_ids):
        raise ValueError('Error: Mapping not complete')

    return mapping


def save_coco_file(new_coco_data, output_dir, coco_filename='merge_coco.json'):
    new_coco_file_path = os.path.join(output_dir, coco_filename)
    
    with open(new_coco_file_path, 'w') as file:
        json.dump(new_coco_data, file, indent=4)
    
    tqdm.tqdm.write(f'COCO file saved at {new_coco_file_path}')
    tqdm.tqdm.write(f'Total images: {len(new_coco_data["images"])}')
    tqdm.tqdm.write(f'Total annotations: {len(new_coco_data["annotations"])}')
    
    # Log dataset statistics
    dataset_counts = {}
    for img in new_coco_data['images']:
        dataset = str(img['dataset'])  # Convert dataset name to string
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    tqdm.tqdm.write('\nImages per dataset:')
    for dataset, count in sorted(dataset_counts.items()):
        tqdm.tqdm.write(f'{dataset}: {count} images')


def get_cat_id_if_on_edge(category_id, xywh_bbox, edge_cat_id, width, height):
    inner_structures_cats = [constants.SEG_TO_COCO_CATEGORIES_MAPPING[cat] for cat in constants.SEG_INNER_CATEGORIES]

    if edge_cat_id is None or category_id == edge_cat_id or category_id in inner_structures_cats:
        return category_id

    if is_xywh_bbox_touching_edges(xywh_bbox, (0, 0, width, height)):
        return edge_cat_id
    
    return category_id
    

def is_xywh_bbox_touching_edges(xywh_bbox, border, threshold=5):
    x, y, w, h = xywh_bbox
    min_x, min_y, max_x, max_y = border
    min_x += threshold
    min_y += threshold
    max_x -= threshold
    max_y -= threshold

    return x <= min_x or y <= min_y or x + w >= max_x or y + h >= max_y


def masks2segments(masks):
    segments = []
    masks = masks.astype(np.uint8)
    for mask in masks:
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contour:
            contour = np.array(contour[np.array([len(x) for x in contour]).argmax()]).reshape(-1, 2)
        else:
            contour = np.zeros((0, 2))
        segments.append(contour.astype(np.float32))

    return segments
        
if __name__ == "__main__":
    main()