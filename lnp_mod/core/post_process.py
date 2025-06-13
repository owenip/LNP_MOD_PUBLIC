# post_process.oy

import cv2
import os
import json
import pandas as pd
import seaborn as sn
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.image as mpimg
import matplotlib.patches as patches
import math
import numpy as np
import tqdm
import lnp_mod.config.constants as constants

from ultralytics.data.utils import LOGGER, colorstr
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from pycocotools.coco import COCO

def post_process(raw_data_json_path, img_dir, output_dir):
    LOGGER.info(f'Post processing started')
    df = create_df_from_raw_data(raw_data_json_path, img_dir)
    df = process_df_data(df)
    create_output_excel(df, output_dir)
    create_output_images(df, output_dir)
    create_output_plots(df, output_dir)
    # create_output_images_from_object_detection(output_dir, img_dir, raw_data_json_path)
    LOGGER.info(f'Post processing finished')


def is_polygon_within_another(polygon1, polygon2, overlap_thresold_percentage=constants.POLYGON_OVERLAP_THRESHOLD_PERC):
    try:
        intersection = polygon1.intersection(polygon2)
        if polygon2.area == 0:
            return False
    
        overlap_percentage = (intersection.area / polygon2.area) * 100
    except Exception as e:
        return False

    return overlap_percentage >= overlap_thresold_percentage


def get_longest_line_in_polygon(polygon):
    try:
        # Check if the convex hull is valid
        if not polygon.convex_hull.is_valid:
            return LineString()

        # Get the convex hull of the polygon
        hull = polygon.convex_hull

        # Get the points of the convex hull
        points = list(hull.exterior.coords)

        # Initialize the maximum distance and the corresponding line
        max_distance = 0
        max_line = LineString()

        # Iterate over all pairs of points
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                # Create a line between the two points
                line = LineString([Point(points[i]), Point(points[j])])

                # Check if the line is inside the polygon
                if polygon.contains(line):
                    # Calculate the distance between the two points
                    distance = Point(points[i]).distance(Point(points[j]))

                    # Update the maximum distance and the corresponding line
                    if distance > max_distance:
                        max_distance = distance
                        max_line = line

        return max_line
    except Exception as e:
        return LineString()

def create_df_from_raw_data(lnp_mod_output_json_path, img_dir):
    LOGGER.info(f'Creating dataframe from data')
    coco = COCO(lnp_mod_output_json_path)

    img_data = coco.loadImgs(coco.getImgIds())
    anns = coco.loadAnns(coco.getAnnIds())

    for index, ann in enumerate(anns):
        # img_data_idx = ann['image_id'] - 1
        # anns[index]['image_name'] = img_data[img_data_idx]['file_name']
        # anns[index]['image_path'] = os.path.join(img_dir, img_data[img_data_idx]['file_name'])
        # anns[index]['image_size'] = [img_data[img_data_idx]['height'], img_data[img_data_idx]['width']]
        # anns[index]['image_width'] = img_data[img_data_idx]['width']
        # anns[index]['image_height'] = img_data[img_data_idx]['height']

        img_data = coco.loadImgs(ann['image_id'])[0]
        anns[index]['image_name'] = img_data['file_name']
        if img_dir is not None:
            anns[index]['image_path'] = os.path.join(img_dir, img_data['file_name'])
        else:
            anns[index]['image_path'] = img_data['file_name']
        anns[index]['image_size'] = [img_data['height'], img_data['width']]
        anns[index]['image_width'] = img_data['width']
        anns[index]['image_height'] = img_data['height']

        if index in anns and 'area' in anns[index]:
            del anns[index]['area']

        # Check if segmentation exists in the annotation
        if 'segmentation' in ann and ann['segmentation'] and len(ann['segmentation']) > 0:
            polygon_points = list(zip(ann['segmentation'][0][::2], ann['segmentation'][0][1::2]))
            anns[index]['polygon'] = Polygon(polygon_points)
        else:
            anns[index]['polygon'] = Polygon()

    df = pd.DataFrame(anns)

    return df


def process_df_data(df):
    LOGGER.info(f'Processing dataframe data')

    # Create all columns first
    df = df.assign(
        post_process_cls=df['category_id'],
        post_process_cls_name=df['category_id'].apply(lambda x: constants.POST_PROCESSING_CATEGORIES[x]),
        area_pxiels=df['polygon'].apply(lambda x: x.area),
    )
    
    # Calculate area_nm2
    df = df.assign(area_nm2=df['area_pxiels'] / constants.PIXELS_IN_NM**2)
    
    # Filter out small areas only for classes not in POST_INNER_CATEGORIES
    mask = (df['post_process_cls'].isin(constants.POST_INNER_CATEGORIES)) | \
           (~df['post_process_cls'].isin(constants.POST_INNER_CATEGORIES) & (df['area_nm2'] >= constants.LNP_AREA_THRESHOLD))
    df = df[mask].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Calculate additional metrics
    df = df.assign(
        radius_nm=df['area_nm2'].apply(lambda x: np.sqrt(x / np.pi)),
        diameter_nm=lambda x: x['radius_nm'] * 2,
        volume_nm3=lambda x: x['radius_nm']**3 * np.pi * 4 / 3,
        x_center=df['bbox'].apply(lambda x: x[0] + x[2] / 2),
        y_center=df['bbox'].apply(lambda x: x[1] + x[3] / 2),
        width_pixels=df['bbox'].apply(lambda x: x[2]),
        height_pixels=df['bbox'].apply(lambda x: x[3]),
        mRNA_area=0.0,
        mRNA_area_nm2=0.0,
        radius_mRNA_area_nm=0.0,
        diameter_mRNA_area_nm=0.0,
        mRNA_volume_nm3=0.0,
        mRNA_per_LNP=0.0,
        mRNA_area_ratio=0.0,
        bleb_ann_id=pd.NA,
        mrna_ann_ids=pd.NA,
        liposomal_lnp_with_mrna_ann_id=pd.NA,
        measured_diameter_LineString=pd.NA,
        measured_diameter_pixels=0,
    )
    
    # Set mRNA area for mRNA class
    mask_mrna = df['post_process_cls'] == constants.SEG_MRNA
    df.loc[mask_mrna, 'mRNA_area_nm2'] = df.loc[mask_mrna, 'area_nm2']

    # Calculate the longest line in the polygon
    pbar = tqdm.tqdm(total=len(df))
    pbar.set_description('Calculating longest line in each found lnp polygon')
    for index, row in df.iterrows():
        pbar.update(1)
        df.loc[index, 'measured_diameter_LineString'] = get_longest_line_in_polygon(row['polygon'])
    pbar.close()

    df['measured_diameter_pixels'] = df['measured_diameter_LineString'].apply(lambda x: x.length if x is not pd.NA else 0)
    df['measured_diameter_nm'] = df['measured_diameter_pixels'] / constants.PIXELS_IN_NM

    ## Bleb with mRNA calculations
    LOGGER.info(f'Analysis Bleb with mRNA data')
    for index, row in df[df['post_process_cls'] == constants.POST_PROCESS_BLEB_WITH_MRNA].iterrows():
        if row['area_nm2'] == 0:
            continue

        bleb_polygon = row['polygon']
        df_mrna = df[
            (df['post_process_cls'] == constants.SEG_MRNA)
            & (df['image_name'] == row['image_name'])
            & df[df['post_process_cls'] == constants.SEG_MRNA]['polygon'].apply(lambda mrna_polygon: is_polygon_within_another(bleb_polygon, mrna_polygon))
        ]

        sum_mrna_area_nm2 = df_mrna['area_nm2'].sum()
        df.loc[index, 'mRNA_area_nm2'] = sum_mrna_area_nm2
        # df.loc[index, 'mRNA_area_ratio'] = sum_mrna_area_nm2 / row['area_nm2'] if row['area_nm2'] != 0 else 0
        # df.loc[index, 'mRNA_per_LNP'] = round(sum_mrna_area_nm2 / constants.MRNA_SIZE, None)

        mean_mRNA_radius = np.sqrt(sum_mrna_area_nm2 / np.pi)
        volumne = (4/3) * np.pi * (mean_mRNA_radius**3)
        df.loc[index, 'mRNA_volume_nm3'] = volumne
        
        df.loc[index, 'mean_mRNA_radius'] = mean_mRNA_radius
        df.loc[index, 'mRNA_per_LNP'] = volumne / constants.MRNA_VOLUME

        # Put the mrna ann ids in the bleb row as a list and separate by comma
        if len(df_mrna) > 0:
            df.loc[index, 'mrna_ann_ids'] = ','.join([str(x) for x in df_mrna['id'].tolist()])

        # Put the bleb ann id in the mrna rows
        df.loc[df_mrna.index, 'bleb_ann_id'] = row['id']
        df.loc[index, 'bleb_ann_id'] = row['id']

    # df = df[~((df['post_process_cls'] == constants.POST_PROCESS_BLEB_WITH_MRNA) & (df['mRNA_area_nm2'] == 0.0))]

    df = calculate_liposomal_lnp(df)
    df = df.round(2)

    return df

def calculate_liposomal_lnp(df):
    LOGGER.info(f'Analysis Liposomal LNP data')
    df = df.assign(
        oil_droplet_area_nm2=0.0,
        oil_droplet_percentage=0.0,
        liposomal_lnp_ann_id=pd.NA,
        oil_droplet_ann_ids=pd.NA,
    )
     
    for index, row in df[df['post_process_cls'] == constants.POST_PROCESS_LIPOSOMAL_LNP].iterrows():
        liposomal_lnp_polygon = row['polygon']
        df_oil_droplet = df[
            (df['post_process_cls'] == constants.POST_PROCESS_OIL_DROPLET)
            & (df['image_name'] == row['image_name'])
            & df[df['post_process_cls'] == constants.POST_PROCESS_OIL_DROPLET]['polygon'].apply(lambda oil_droplet_polygon: is_polygon_within_another(liposomal_lnp_polygon, oil_droplet_polygon))
        ]

        df_mrna = df[
            (df['post_process_cls'] == constants.POST_PROCESS_MRNA)
            & (df['image_name'] == row['image_name'])
            & df[df['post_process_cls'] == constants.POST_PROCESS_MRNA]['polygon'].apply(lambda mrna_polygon: is_polygon_within_another(liposomal_lnp_polygon, mrna_polygon))
        ]

        sum_oil_droplet_area_nm2 = df_oil_droplet['area_nm2'].sum()
        df.loc[index, 'oil_droplet_area_nm2'] = round(sum_oil_droplet_area_nm2, 2)
        df.loc[index, 'oil_droplet_percentage'] = round(sum_oil_droplet_area_nm2 / row['area_nm2'] * 100, 2) if row['area_nm2'] != 0 else 0
        
        sum_mrna_area_nm2 = df_mrna['area_nm2'].sum()
        df.loc[index, 'mRNA_area_nm2'] = sum_mrna_area_nm2
        df.loc[index, 'mRNA_area_ratio'] = sum_mrna_area_nm2 / row['area_nm2'] if row['area_nm2'] != 0 else 0
        df.loc[index, 'mRNA_per_LNP'] = sum_mrna_area_nm2 / constants.MRNA_SIZE

        if len(df_oil_droplet) > 0:
            df.loc[index, 'oil_droplet_ann_ids'] = ','.join([str(x) for x in df_oil_droplet['id'].tolist()])
            
            if len(df_mrna) == 0:
                df.loc[df_oil_droplet.index, 'liposomal_lnp_ann_id'] = row['id']
                df.loc[index, 'liposomal_lnp_ann_id'] = row['id']
            else:
                df.loc[df_oil_droplet.index, 'liposomal_lnp_with_mrna_ann_id'] = row['id']
                df.loc[index, 'liposomal_lnp_with_mrna_ann_id'] = row['id']

        if len(df_mrna) > 0:
            df.loc[index, 'mrna_ann_ids'] = ','.join([str(x) for x in df_mrna['id'].tolist()])
            df.loc[index, 'post_process_cls'] = constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA
            df.loc[index, 'post_process_cls_name'] = constants.POST_PROCESSING_CATEGORIES[constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA]
            
            df.loc[df_mrna.index, 'liposomal_lnp_with_mrna_ann_id'] = row['id']
            df.loc[index, 'liposomal_lnp_with_mrna_ann_id'] = row['id']

    return df


def create_output_excel(df, output_dir):
    LOGGER.info(f'Creating excel file')
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'mRNA_area_nm2',
        'mRNA_per_LNP',
        'bleb_ann_id',
        'mrna_ann_ids',
        'measured_diameter_nm',
    ]
    df_result = df.loc[:, output_columns]
    df_result = df_result.rename(columns={'post_process_cls_name': 'class_name'})
    df_result = df_result.round(2)

    df_summary = create_image_summary(df)

    excel_file_path = os.path.join(output_dir, 'result.xlsx')
    with pd.ExcelWriter(excel_file_path, mode='w') as writer:
        df_result.to_excel(writer, sheet_name='found_lnps', index=False)
        df_summary.to_excel(writer, sheet_name='image_summary', index=False)

    append_class_summaries_to_excel(df, excel_file_path)

def create_image_summary(df):
    include_classes = [
        constants.POST_PROCESS_BLEB_WITH_MRNA,
        constants.POST_PROCESS_OIL_CORE,
        constants.POST_PROCESS_LIPOSOMAL_LNP,
        constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA,
        constants.POST_PROCESS_LIPOSOME,
    ]

     # Exclude mRNA, not fully visible LNP, and oil droplet
    df_valid_lnp = df[df['post_process_cls'].isin(include_classes)]

    df_summary = pd.DataFrame({
        'image_name': df_valid_lnp['image_name'].unique(),
        'Count': df_valid_lnp.groupby('image_name')['post_process_cls'].count(),
    })

    for cls in include_classes:
        cls_name = constants.POST_PROCESSING_CATEGORIES[cls]
        df_summary[cls_name] = df_valid_lnp[df_valid_lnp['post_process_cls'] == cls].groupby('image_name')['post_process_cls'].count()

    df_summary.fillna(0, inplace=True)
    # Add percentage columns
    for cls in include_classes:
        cls_name = constants.POST_PROCESSING_CATEGORIES[cls]
        df_summary[f'{cls_name}%'] = round(df_summary[cls_name] / df_summary['Count'] * 100, 2)

    # Add total row
    row_total = df_summary.sum()
    row_total['image_name'] = 'Total'
    for cls in include_classes:
        cls_name = constants.POST_PROCESSING_CATEGORIES[cls]
        row_total[f'{cls_name}%'] = round(row_total[cls_name] / row_total['Count'] * 100, 2)
    
    df_summary = pd.concat([df_summary, pd.DataFrame(row_total).T], ignore_index=True)

    return df_summary


def append_class_summaries_to_excel(df, excel_path):
    sheet_name_function_mapping = {
        'Bleb with mRNA': create_bleb_with_mrna_summary,
        'Oil Core': create_oil_core_summary,
        'Liposomal LNP': create_liposomal_lnp_with_oil_droplet_summary,
        'Liposomal LNP with mRNA': create_liposomal_lnp_with_mrna_summary,
        'Liposome': create_liposome_summary,
        'mRNA not inside LNP': create_outline_mrna_summary,
        'Oil Droplet not inside LNP': create_ouliner_oil_droplet_summary,
    }

    for sheet_name in sheet_name_function_mapping:
        df_class_summary = sheet_name_function_mapping[sheet_name](df)
        df_class_summary = df_class_summary.round(2)
        df_class_summary.rename(
            columns={'post_process_cls': 'class_id', 'post_process_cls_name': 'class_name'},
            inplace=True
        )

        if not os.path.exists(excel_path):
            mode = 'w'
        else:
            mode = 'a'

        with pd.ExcelWriter(excel_path, mode=mode) as writer:
            df_class_summary.to_excel(writer, sheet_name=sheet_name, index=False)


def create_class_summary(df, class_names):
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'measured_diameter_nm',
    ]

    df_class_summary = df[df['post_process_cls'].isin(class_names)]
    df_class_summary = df_class_summary.loc[:, output_columns]

    df_mean = df_class_summary.groupby('post_process_cls_name').mean(numeric_only=True).reset_index()
    df_mean['id'] = 'Average'
    df_mean['image_name'] = pd.NA

    df_std = df_class_summary.groupby('post_process_cls_name').std(numeric_only=True).reset_index()
    df_std['id'] = 'Standard Deviation'
    df_std['image_name'] = pd.NA

    df_class_summary = pd.concat([df_class_summary, df_mean], ignore_index=True)
    df_class_summary = pd.concat([df_class_summary, df_std], ignore_index=True)

    return df_class_summary


def create_bleb_with_mrna_summary(df):
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'measured_diameter_nm',
        'mRNA_area_nm2',
        'mRNA_area_ratio',
        'mRNA_per_LNP',
        'bleb_ann_id',
        'mrna_ann_ids'
    ]

    df_bleb_with_mrna = df[
        df['post_process_cls'].isin([constants.POST_PROCESS_BLEB_WITH_MRNA, constants.POST_PROCESS_MRNA])
        & df['bleb_ann_id'].notna()
    ]

    df_bleb_with_mrna = df_bleb_with_mrna.loc[:, output_columns]
    df_bleb_with_mrna = df_bleb_with_mrna.sort_values(by=['bleb_ann_id', 'mrna_ann_ids'], kind='mergesort')

    df_mean = df_bleb_with_mrna.groupby('post_process_cls_name').mean(numeric_only=True).reset_index()
    df_mean['id'] = 'Average'
    df_mean['image_name'] = pd.NA
    df_mean['bleb_ann_id'] = pd.NA
    df_mean['mrna_ann_ids'] = pd.NA

    df_std = df_bleb_with_mrna.groupby('post_process_cls_name').std(numeric_only=True).reset_index()
    df_std['id'] = 'Standard Deviation'
    df_std['image_name'] = pd.NA
    df_std['bleb_ann_id'] = pd.NA
    df_std['mrna_ann_ids'] = pd.NA
    
    df_bleb_with_mrna = pd.concat([df_bleb_with_mrna, df_mean], ignore_index=True)
    df_bleb_with_mrna = pd.concat([df_bleb_with_mrna, df_std], ignore_index=True)

    return df_bleb_with_mrna

def create_oil_core_summary(df):
    return create_class_summary(df, [constants.POST_PROCESS_OIL_CORE])

def create_liposome_summary(df):
    return create_class_summary(df, [constants.POST_PROCESS_LIPOSOME])

def create_liposomal_lnp_only_summary(df):
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'measured_diameter_nm',
    ]
        
    df_liposomal_lnp_only = df[df['post_process_cls'].isin([constants.POST_PROCESS_LIPOSOMAL_LNP])]
    df_liposomal_lnp_only = df_liposomal_lnp_only.loc[:, output_columns]

    df_mean = df_liposomal_lnp_only.groupby('post_process_cls_name').mean(numeric_only=True).reset_index()
    df_mean['id'] = 'Average'
    df_mean['image_name'] = pd.NA

    df_std = df_liposomal_lnp_only.groupby('post_process_cls_name').std(numeric_only=True).reset_index()
    df_std['id'] = 'Standard Deviation'
    df_std['image_name'] = pd.NA

    df_liposomal_lnp_only = pd.concat([df_liposomal_lnp_only, df_mean], ignore_index=True)
    df_liposomal_lnp_only = pd.concat([df_liposomal_lnp_only, df_std], ignore_index=True)

    return df_liposomal_lnp_only

def create_liposomal_lnp_with_oil_droplet_summary(df):
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'measured_diameter_nm',
        'liposomal_lnp_ann_id',
        'oil_droplet_ann_ids'
    ]
        
    df_liposomal_lnp_with_oil_droplet = df[
        df['post_process_cls'].isin([constants.POST_PROCESS_LIPOSOMAL_LNP, constants.POST_PROCESS_OIL_DROPLET])
        & df['liposomal_lnp_ann_id'].notna()
    ].copy()

    # iterate over the rows and remove the oil droplet row if the corresponding liposomal_lnp_ann_id of the post_process_cls is not liposomal_lnp
    for index, row in df_liposomal_lnp_with_oil_droplet.iterrows():
        if row['post_process_cls'] == constants.POST_PROCESS_OIL_DROPLET:
            if df.loc[df['id'] == row['liposomal_lnp_ann_id'], 'post_process_cls'].values[0] != constants.POST_PROCESS_LIPOSOMAL_LNP:
                df_liposomal_lnp_with_oil_droplet.drop(index, inplace=True)

    df_liposomal_lnp_with_oil_droplet = df_liposomal_lnp_with_oil_droplet.loc[:, output_columns]

    df_mean = df_liposomal_lnp_with_oil_droplet.groupby('post_process_cls_name').mean(numeric_only=True).reset_index()
    df_mean['id'] = 'Average'
    df_mean['image_name'] = pd.NA

    df_std = df_liposomal_lnp_with_oil_droplet.groupby('post_process_cls_name').std(numeric_only=True).reset_index()
    df_std['id'] = 'Standard Deviation'
    df_std['image_name'] = pd.NA

    df_liposomal_lnp_with_oil_droplet = pd.concat([df_liposomal_lnp_with_oil_droplet, df_mean], ignore_index=True)
    df_liposomal_lnp_with_oil_droplet = pd.concat([df_liposomal_lnp_with_oil_droplet, df_std], ignore_index=True)

    df_liposomal_lnp_with_oil_droplet = df_liposomal_lnp_with_oil_droplet.sort_values(by=['liposomal_lnp_ann_id', 'oil_droplet_ann_ids'], kind='mergesort')

    return df_liposomal_lnp_with_oil_droplet

def create_ouliner_oil_droplet_summary(df):
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'measured_diameter_nm',
    ]

    df_outliner_oil_droplet = df[
        df['post_process_cls'].isin([constants.POST_PROCESS_OIL_DROPLET])
            & df['liposomal_lnp_ann_id'].isna()
    ]

    df_mean = df_outliner_oil_droplet.groupby('post_process_cls_name').mean(numeric_only=True).reset_index()
    df_mean['id'] = 'Average'
    df_mean['image_name'] = pd.NA

    df_std = df_outliner_oil_droplet.groupby('post_process_cls_name').std(numeric_only=True).reset_index()
    df_std['id'] = 'Standard Deviation'
    df_std['image_name'] = pd.NA

    df_outliner_oil_droplet = pd.concat([df_outliner_oil_droplet, df_mean], ignore_index=True)
    df_outliner_oil_droplet = pd.concat([df_outliner_oil_droplet, df_std], ignore_index=True)

    df_outliner_oil_droplet = df_outliner_oil_droplet.loc[:, output_columns]

    return df_outliner_oil_droplet

def create_liposomal_lnp_with_mrna_summary(df):
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'measured_diameter_nm',
        'mRNA_area_nm2',
        'mRNA_area_ratio',
        'mRNA_per_LNP',
        'liposomal_lnp_with_mrna_ann_id',
        'oil_droplet_ann_ids',
        'mrna_ann_ids',
    ]
    
    df_summary = df[
        df['post_process_cls'].isin([constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA, constants.POST_PROCESS_OIL_DROPLET, constants.POST_PROCESS_MRNA])
        & df['liposomal_lnp_with_mrna_ann_id'].notna()
    ]

    df_mean = df_summary.groupby('post_process_cls_name').mean(numeric_only=True).reset_index()
    df_mean['id'] = 'Average'
    df_mean['image_name'] = pd.NA
    df_mean['liposomal_lnp_with_mrna_ann_id'] = pd.NA
    df_mean['mrna_ann_ids'] = pd.NA

    df_std = df_summary.groupby('post_process_cls_name').std(numeric_only=True).reset_index()
    df_std['id'] = 'Standard Deviation'
    df_std['image_name'] = pd.NA
    df_std['liposomal_lnp_with_mrna_ann_id'] = pd.NA
    df_std['mrna_ann_ids'] = pd.NA

    df_summary = pd.concat([df_summary, df_mean], ignore_index=True)
    df_summary = pd.concat([df_summary, df_std], ignore_index=True)
    
    df_summary = df_summary.loc[:, output_columns]
    df_summary = df_summary.sort_values(by=['liposomal_lnp_with_mrna_ann_id','oil_droplet_ann_ids', 'mrna_ann_ids'], kind='mergesort')

    return df_summary


def create_outline_mrna_summary(df):
    output_columns = [
        'id',
        'image_name',
        'post_process_cls_name',
        'area_nm2',
        'radius_nm',
        'diameter_nm',
        'volume_nm3',
        'measured_diameter_nm',
        'mRNA_area_nm2',
    ]

    df_summary = df[
        df['post_process_cls'].isin([constants.POST_PROCESS_MRNA])
        & df['bleb_ann_id'].isna()
        & df['liposomal_lnp_with_mrna_ann_id'].isna()
    ]

    df_mean = df_summary.groupby('post_process_cls_name').mean(numeric_only=True).reset_index()
    df_mean['id'] = 'Average'
    df_mean['image_name'] = pd.NA

    df_std = df_summary.groupby('post_process_cls_name').std(numeric_only=True).reset_index()
    df_std['id'] = 'Standard Deviation'
    df_std['image_name'] = pd.NA

    df_summary = pd.concat([df_summary, df_mean], ignore_index=True)
    df_summary = pd.concat([df_summary, df_std], ignore_index=True)

    df_summary = df_summary.loc[:, output_columns]

    return df_summary


def create_output_images(df, output_dir, mask_only=False):
    classes = [
        constants.POST_PROCESS_BLEB_WITH_MRNA,
        constants.POST_PROCESS_MRNA,
        constants.POST_PROCESS_OIL_CORE,
        constants.POST_PROCESS_OTHER_LNP,
        constants.POST_PROCESS_NOT_FULLY_VISIBLE_LNP,
        constants.POST_PROCESS_LIPOSOMAL_LNP,
        constants.POST_PROCESS_OIL_DROPLET,
        constants.POST_PROCESS_LIPOSOME,
        constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA,
    ]
    
    df_unique_images = df[df['post_process_cls'].isin(classes)]['image_name'].unique()
    # Create figure legend
    legend_elements = [patches.Patch(facecolor=constants.POST_PROCESS_LABEL_COLORS[cls], edgecolor='none', label=constants.POST_PROCESSING_CATEGORIES[cls]) for cls in classes]

    total_images = len(df_unique_images)
    # Loop through each unique image
    for image_name in df_unique_images:
        LOGGER.info(f'Generating images with annotations: {image_name}')

        # Get the rows in the df that are for the current image
        image_df = df.loc[df['image_name'] == image_name]

        # Get the image size
        image_size = image_df['image_size'].iloc[0]

        # Get the image path
        image_path = image_df['image_path'].iloc[0]

        # Render the image with matplotlib
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(image_size[1] / 100, image_size[0] / 100))
        ax.imshow(img)

        sorted_df = image_df[image_df['post_process_cls'].isin(classes)].sort_values('area_nm2', ascending=False)
        # Loop through each row in the image df
        for i, row in sorted_df.iterrows():
            cls = row['post_process_cls']
            color = constants.POST_PROCESS_LABEL_COLORS[cls]
            # Get the bbox coordinates
            bbox = row['bbox']
            x1, y1, w, h = bbox
            # Plot the bbox and ID
            if not mask_only:
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            # Add LNP ID as label
            ax.text(x1 + w, y1, f'#{row["id"]}', fontsize=24, color='white', ha='right', va='top')
            
            # Get the segmentation coordinates
            segmentations_polygon = row['polygon']
            # Plot the segmentation
            if cls != constants.POST_PROCESS_NOT_FULLY_VISIBLE_LNP and not segmentations_polygon.is_empty:
                patch = patches.Polygon(list(segmentations_polygon.exterior.coords), edgecolor=color, fill=True, facecolor=color, alpha=0.3)
                ax.add_patch(patch)
                # Plot the measured_diameter_LineString
                if row['measured_diameter_LineString'] is not np.NAN and row['measured_diameter_LineString'] is not pd.NA:
                    line = row['measured_diameter_LineString']
                    # ax.plot(*line.xy, color='white', linewidth=1)

        ax.legend(handles=legend_elements, loc='lower left')
        ax.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,  hspace = 0, wspace = 0)
        plt.margins(0,0)
        # Save the rendered image
        plt.savefig(os.path.join(output_dir, image_name.replace('.tif', '.png')), dpi=100)
        plt.close()


def create_output_images_from_object_detection(output_dir, img_dir, coco_path):
    LOGGER.info(f'Creating output images from object detection')
    # od_output_json_path = os.path.join(output_dir, 'od_output.json')
    od_df = create_df_from_raw_data(coco_path, img_dir)

    od_images_dir = os.path.join(output_dir, 'od_images')
    os.makedirs(od_images_dir, exist_ok=True)
    
    # for each unique image name in od_df, create a new image with the bounding boxes
    for image_name in od_df['image_name'].unique():
        img_path = os.path.join(img_dir, image_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(od_df['image_width'].unique()[0] / 100, od_df['image_height'].unique()[0] / 100))
        ax.imshow(img)
        ax.axis('off')
        fig.tight_layout()

        # Plot the bounding boxes
        for _, row in od_df.iterrows():
            if row['image_name'] != image_name:
                continue

            bbox = row['bbox']
            x1, y1, w, h = bbox
            color = constants.POST_PROCESS_LABEL_COLORS[row['category_id']]
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # save the plot
        plt.savefig(os.path.join(od_images_dir, f'{image_name}_od.png'))
        plt.close()



def create_output_plots(df, output_dir):
    LOGGER.info(f'Plotting graphs')
    classes_to_plot = [
        constants.POST_PROCESS_BLEB_WITH_MRNA,
        # constants.POST_PROCESS_MRNA,
        constants.POST_PROCESS_OIL_CORE,
        constants.POST_PROCESS_OTHER_LNP,
        constants.POST_PROCESS_LIPOSOMAL_LNP,
        # constants.POST_PROCESS_OIL_DROPLET,
        constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA,
        constants.POST_PROCESS_LIPOSOME,
    ]

    figure_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    create_count_plots(df, figure_dir, classes_to_plot)
    create_area_nm2_plot(df, figure_dir)
    create_area_hist_plots(df, figure_dir)
    create_diameter_hist_plots(df, figure_dir)
    


def create_count_plots(df, output_dir, post_process_classes_to_plot=constants.POST_PROCESSING_CATEGORIES.keys()):
    palette = get_palette(post_process_classes_to_plot, constants.POST_PROCESS_LABEL_COLORS)

    df_count = df[df['post_process_cls'].isin(post_process_classes_to_plot)].groupby('post_process_cls').size().reset_index(name='count')
    df_count['post_process_cls_name'] = df_count['post_process_cls'].apply(lambda x: constants.POST_PROCESSING_CATEGORIES[x])

    plt.figure()
    count_plot = sn.barplot(x='post_process_cls_name', y='count', data=df_count, palette=palette)
    for index, row in df_count.iterrows():
        count_plot.annotate(row['count'], (index, row['count']), ha='center', va='bottom', textcoords='offset points', xytext=(0, -2))

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.suptitle('Count of all found LNPs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'found_lnps_count.png'), dpi=100)
    plt.close()

    df_count['percentage'] = round(df_count['count'] / df_count['count'].sum() * 100, 2)
    plt.figure()
    percentage_plot = sn.barplot(x='post_process_cls_name', y='percentage', data=df_count, palette=palette)
    # Add data points at the bars
    for index, row in df_count.iterrows():
        percentage_plot.annotate(f"{row['percentage']}%", (index, row['percentage']), ha='center', va='bottom', textcoords='offset points', xytext=(0, 0))

    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.suptitle('Percentage of all found LNPs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'found_lnps_percentage.png'), dpi=100)
    plt.close()
    

def create_area_hist_plots(df, output_dir):
    x_label = 'Area (nm^2)'
    y_label = 'Frequency'

    class_fig_title_mapping = {
        constants.POST_PROCESS_BLEB_WITH_MRNA: f'Area of {constants.POST_PROCESSING_CATEGORIES[constants.POST_PROCESS_BLEB_WITH_MRNA]}',
        constants.POST_PROCESS_OIL_CORE: f'Area of {constants.POST_PROCESSING_CATEGORIES[constants.POST_PROCESS_OIL_CORE]}',
        constants.POST_PROCESS_LIPOSOMAL_LNP: f'Area of {constants.POST_PROCESSING_CATEGORIES[constants.POST_PROCESS_LIPOSOMAL_LNP]}',
        constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA: f'Area of {constants.POST_PROCESSING_CATEGORIES[constants.POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA]}',
        constants.POST_PROCESS_LIPOSOME: f'Area of {constants.POST_PROCESSING_CATEGORIES[constants.POST_PROCESS_LIPOSOME]}',
    }

    for cls, fig_title in class_fig_title_mapping.items():
        create_hist_plot(
            df[df['post_process_cls'] == cls]['area_nm2'],
            os.path.join(output_dir, f'{constants.POST_PROCESSING_CATEGORIES[cls].replace(" ", "_")}_area_hist.png'),
            fig_title,
            x_label,
            y_label
        )


def create_area_nm2_plot(df, output_dir, post_process_classes_to_plot=constants.POST_PROCESSING_CATEGORIES.keys()):
    palette = get_palette(post_process_classes_to_plot, constants.POST_PROCESS_LABEL_COLORS)
    size_plot = sn.boxplot(x='post_process_cls_name', y='area_nm2', hue='post_process_cls_name', data=df[df['post_process_cls'].isin(post_process_classes_to_plot)].sort_values('post_process_cls',ascending=True), palette=palette)
    plt.xlabel('Class')
    plt.ylabel('Cross Sertional Area (nm^2)')
    plt.suptitle('Size of all found LNPs')
    plt.xticks(rotation=45)
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(output_dir, 'found_lnps_size_area_nm2.png'), dpi=100)
    plt.close()


def create_diameter_hist_plots(df, output_dir, post_process_classes_to_plot=constants.POST_PROCESSING_CATEGORIES.keys()):
    x_label = 'Diameter (nm)'
    y_label = 'Frequency'

    for cls in post_process_classes_to_plot:
        create_hist_plot(
            df[df['post_process_cls'] == cls]['diameter_nm'],
            os.path.join(output_dir, f'{constants.POST_PROCESSING_CATEGORIES[cls].replace(" ", "_")}_diameter_hist.png'),
            f'Diameter of {constants.POST_PROCESSING_CATEGORIES[cls]}',
            x_label,
            y_label
        )

def create_hist_plot(data, fig_path, title, x_label, y_label):
    hist_plot = sn.histplot(data, kde=True, bins='auto')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=100)
    plt.close()

def create_w_h_plot(df, output_dir, post_process_classes_to_plot=constants.POST_PROCESSING_CATEGORIES.keys()):
    palette = get_palette(post_process_classes_to_plot, constants.POST_PROCESS_LABEL_COLORS)

    hist = sn.histplot(
        df[df['post_process_cls'].isin(post_process_classes_to_plot)],
        x='width_pixels',
        y='height_pixels',
        hue='post_process_cls_name',
        bins='auto',
        palette=palette,
    )
    sn.move_legend(hist, "upper left", bbox_to_anchor=(1, 1), title='Class')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.suptitle('Width and Height of all found LNPs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'found_lnps_w_h_hist.png'), dpi=300)
    plt.close()    
    
    pair_plot = sn.pairplot(
        df[df['post_process_cls'].isin(post_process_classes_to_plot)][['width_pixels', 'height_pixels']],
        corner=True,
        diag_kind='auto',
        kind='hist',
        diag_kws=dict(bins=50),
        plot_kws=dict(pmax=0.9)
    )
    plt.tight_layout()
    plt.suptitle(f'Xorrelogram of all class')
    plt.savefig(os.path.join(output_dir, 'pair_plot_all_class.png'), dpi=100)
    plt.close()

    for cls in post_process_classes_to_plot:
        pair_plot = sn.pairplot(
            df[df['post_process_cls']==cls][['width_pixels', 'height_pixels']],
            corner=True,
            diag_kind='auto',
            kind='hist',
            diag_kws={'bins': 'auto'},
            plot_kws=dict(pmax=0.9)
        )
        plt.tight_layout()
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.suptitle(f'Xorrelogram of {constants.POST_PROCESSING_CATEGORIES[cls]}')
        plt.savefig(os.path.join(output_dir, f'pair_plot_{constants.POST_PROCESSING_CATEGORIES[cls].replace(" ", "_")}.png'), dpi=100)
        plt.close()


def get_palette(classes, palette):
    return [palette[cls] for cls in classes]

def convert_df_to_coco(df, output_dir):
    LOGGER.info(f'Creating COCO JSON file')
    # Create the COCO object
    coco = COCO()
    # Create the info object
    info = {
        "description": "LNP dataset",
    }
    coco.dataset = info
    # Create the categories object
    categories = []
    for cls, cls_name in constants.SEGMENTATION_CATEGORIES.items():
        categories.append({
            "id": cls + 1,
            "name": cls_name
        })

    coco.dataset['categories'] = categories
    # Create the images object
    images = []
    images_name_id_mapping = {}
    for image_name in df['image_name'].unique():
        image_df = df[df['image_name'] == image_name]
        image_path = image_df['image_path'].iloc[0]
        image = {
            "id": len(images) + 1,
            "file_name": image_name,
            "height": int(image_df['image_size'].iloc[0][0]),  # Convert int64 to int
            "width": int(image_df['image_size'].iloc[0][1])  # Convert int64 to int
        }
        images_name_id_mapping[image_name] = image['id']
        images.append(image)

    coco.dataset['images'] = images

    # Create the annotations object
    annotations = []
    for i, row in df.iterrows():
        simplfy_polygon = row['polygon'].simplify(0.8, preserve_topology=True)
        annotation = {
            "id": row['id'],
            "image_id": images_name_id_mapping[row['image_name']],
            "category_id": row['segmentation_cls'] + 1,
            "segmentation": [[val for xy in simplfy_polygon.exterior.coords for val in xy]],
            "area": row['area_pxiels'],
            "bbox": row['bbox'],
            "iscrowd": 0
        }
        annotations.append(annotation)

    coco.dataset['annotations'] = annotations

    with open(os.path.join(output_dir, 'coco.json'), 'w') as f:
        json.dump(coco.dataset, f)

    LOGGER.info(f'Created COCO JSON file')
