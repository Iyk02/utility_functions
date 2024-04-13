#!/usr/bin/python3

""" Converts coco json annotation files to masks
"""

import json
import numpy as np
import skimage
import os
import matplotlib.pyplot as plt


def create_mask(image_info, annotations, output_folder):
    """ Create masks from json annotation files """
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint16)

    object_number = 1

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            for seg in ann['segmentation']:
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc] = object_number
                object_number += 1

    mask_path = os.path.join(output_folder, image_info['file_name'].replace('.png', '_mask.png'))
    plt.imsave(mask_path, mask_np)

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")


def main(json_file, mask_output_folder):
    """ Main function """
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)

    for img in images:
        create_mask(img, annotations, mask_output_folder)


if __name__ == '__main__':
    json_file = 'annotations/instances_default.json'
    mask_output_folder = 'masks'
    main(json_file, mask_output_folder)
