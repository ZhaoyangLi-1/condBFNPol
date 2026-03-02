#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : extract_keypoints.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/16/2025
#
# Distributed under terms of the MIT license.

"""

"""
import tap
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch

from keypoint_utils import overlay_mask_simple, load_data, show_images
from keypoint_tracker import Tracker


class MaskPointPicker:
    """
    MaskPicker is used to pick the mask to sample keypoints or pick the keypoints directly.
    """
    def __init__(self, config):
        # load SAM model
        self.config = config
        self.rgb_im = None
        self.pcd = None
        device = config.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", self.config.sam2_ckpt_path).to(device=device)
        self.predictor = SAM2ImagePredictor(sam)
        # 1024 x 1024 is the input size for SAM pretrained model
        self.device = torch.device(device)

        self.seeding_point = []
        self.image = None

    def get_seeding_point(self, rgb_image):
        self.seeding_point = []
        self.image = rgb_image
        cv2.imshow('image', rgb_image[..., ::-1])
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.seeding_point

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Coordinates: ({x}, {y})")
            self.seeding_point.append((x, y))
            image_show = self.image.copy()
            cv2.circle(image_show, (x, y), 3, (0, 0, 255), -1)
            for prev_x, prev_y in self.seeding_point:
                cv2.circle(image_show, (prev_x, prev_y), 3, (0, 255, 0), -1)
            cv2.imshow('image', image_show[..., ::-1])

    def select_mask_from_point_query(self, seeding_point, input_label=None) -> list[np.ndarray]:
        """
        Query SAM with point prompts.
        """
        if input_label is None:
            input_label = np.ones(len(seeding_point))
        with torch.no_grad():
            self.predictor.set_image(self.image)
            masks, scores, logits = self.predictor.predict(
                point_coords=seeding_point,
                point_labels=input_label,
                multimask_output=True,
            )
        mask_list = []
        for (mask, score) in zip(masks, scores):
            mask_list.append((mask, score))
        mask_list.sort(key=lambda x: x[1], reverse=True)
        fig, axes = plt.subplots(1, len(mask_list), figsize=(20, 20))
        for i, (mask, score) in enumerate(mask_list):
            axes[i].imshow(overlay_mask_simple(self.image, mask))
            axes[i].set_title(f'Mask {i+1:02d}. Score: {score:.3f}')
            axes[i].axis('off')
        plt.show()
        plt.close()
        selected_mask_id = input('Select mask number to keep: ')
        selected_mask = mask_list[int(selected_mask_id)-1]
        return selected_mask[0]

    def select_keypoints(self, rgb_image):
        self.seeding_point = []
        self.image = rgb_image
        cv2.imshow('image', rgb_image[..., ::-1])
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.seeding_point


class MaskPickerConfig(tap.Tap):
    """
    MaskPickerConfig is used to configure the MaskPicker.
    """
    # path to the checkpoint of SAM model
    sam2_ckpt_path: str = './sam2.1_hiera_large.pt'
    min_area_percentage: float = .0001

    device: str = 'cuda'


def test_keypoint_selection():
    mask_picker = MaskPointPicker(MaskPickerConfig())
    keypoint_tracker = Tracker()
    data = load_data('trajectory_20250407-171604_replay.pkl')
    im = data[0]['mount2']['rgb_im']

    # # Run auto keypoint selection
    # seeding_points = mask_picker.get_seeding_point(im)
    # selected_mask = mask_picker.select_mask_from_point_query(seeding_points)
    # show_images(im, selected_mask)

    # Manual keypoint selection
    selected_keypoints = mask_picker.select_keypoints(im)
    print(selected_keypoints)
    keypoint_tracker.initialize(im, np.array(selected_keypoints))
    keypoint_tracker.track_offline([im for _ in range(10)], save_name='tracked.mp4')


def main(data_folder):
    pkl_filenames = glob.glob(data_folder + '/**_replay.pkl')
    print(pkl_filenames)

    mask_picker = MaskPointPicker(MaskPickerConfig())
    keypoint_tracker = Tracker()

    keypoint_records = []

    for pkl_filename_full in pkl_filenames:
        data = load_data(pkl_filename_full)
        pkl_filename = pkl_filename_full.split('/')[-1]
        print(f"Processing {pkl_filename}")
        image_sequence = [step_data['mount2']['rgb_im'] for step_data in data]
        init_im = image_sequence[0]
        selected_keypoints = np.array(mask_picker.select_keypoints(init_im))
        print(selected_keypoints)
        keypoint_tracker.initialize(init_im, selected_keypoints)
        tracked_keypoints_xy, tracked_keypoints_visibility = keypoint_tracker.track_offline(image_sequence[1:], save_name=f'tracked_{pkl_filename[:-4]}.mp4')
        keypoint_records.append({
            'record_name': pkl_filename,
            'keypoints_xy': np.concatenate((selected_keypoints[np.newaxis, ...],tracked_keypoints_xy)),
            'keypoints_visibility': np.concatenate((np.ones((1, selected_keypoints.shape[0]), dtype=bool), tracked_keypoints_visibility)),
        })

    np.savez('keypoint_records.npz', keypoint_records)


if __name__ == '__main__':
    main('../../../hat/data/real/drawer_250407/')
    # test_keypoint_selection()


