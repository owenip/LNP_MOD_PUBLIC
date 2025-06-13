# Description: This script takes a COCO file and images directory as input and creates an excel file and a folder with images for the given COCO file.

import argparse
import os
import pandas as pd
import json
import lnp_mod.config.constants as constants
import lnp_mod.core.post_process as post_process
import lnp_mod.utils.utils as utils

from ultralytics.utils import LOGGER, colorstr


def main():
    # Take args from cml --coco-file --output-dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", help="COCO file path")
    parser.add_argument("--img_dir", help="Images directory path")
    parser.add_argument("-o", "--output_dir", help="Output directory path", default=None)

    args = parser.parse_args()

    coco_file = args.coco
    img_dir = args.img_dir
    img_dir = os.path.join(img_dir, '')
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = utils.create_output_folder(img_dir)
    else:
        output_dir = os.path.join(output_dir, '')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if not os.path.exists(coco_file):
        LOGGER.error(f"{colorstr('ERROR:')} COCO file does not exist. Please provide a valid COCO file.")
        exit()

    if not os.path.exists(img_dir):
        LOGGER.error(f"{colorstr('ERROR:')} Images directory does not exist. Please provide a valid images directory.")
        exit()

    # Check if img_dir is empty
    if len(os.listdir(img_dir)) == 0:
        LOGGER.error(f"{colorstr('ERROR:')} Images directory is empty. Please provide a valid images directory.")
        exit()

    # df = post_process.process_df_data(df)
    # post_process.create_output_excel(df, output_dir)
    # post_process.create_output_images(df, output_dir)

    post_process.post_process(coco_file, img_dir, output_dir)

    # post_process.create_output_images_from_object_detection(output_dir, img_dir, coco_file)

if __name__ == "__main__":
    main()
