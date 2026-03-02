#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : keypoint_utils.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/16/2025
#
# Distributed under terms of the MIT license.

"""

"""
import pickle
import numpy as np
from matplotlib import pyplot as plt


def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def overlay_mask_simple(rgb_im, mask: np.ndarray, colors=None, mask_alpha=.5):
    if rgb_im.max() > 2:
        rgb_im = rgb_im.astype(np.float32) / 255.
    if colors is None:
        colors = np.array([1, 0, 0])
    return (rgb_im * (1 - mask_alpha) + mask[..., np.newaxis] * colors * mask_alpha).copy()


def show_images(im, selected_mask):
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    axes[0].imshow(im)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(overlay_mask_simple(im, selected_mask))
    axes[1].set_title('Overlayed Image')
    axes[1].axis('off')
    plt.show()
    plt.close()
