import cv2
import numpy as np

from utils import (
    get_video_files,
    release_video_files,
    write_video,
    load_entire_video,
    scale_matrix_0_to_255,
    apply_mask_on_color_frame
)


def background_substraction(input_video_path, output_video_path):
    # Read input video
    cap, out, w, h, fps = get_video_files(input_video_path, output_video_path, isColor=True)
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    _, prev = cap.read()

    frames_bgr = load_entire_video(cap, color_space='bgr')
    frames_hsv = load_entire_video(cap, color_space='hsv')

    "Find medians in colors space "
    # medians_frame_bgr = np.median(frames_bgr, axis=0)
    # medians_frame_b, medians_frame_g, medians_frame_r = cv2.split(medians_frame_bgr)
    # medians_frame_hsv = np.median(frames_hsv, axis=0)
    # medians_frame_h, medians_frame_s, medians_frame_v = cv2.split(medians_frame_hsv)

    medians_frame_b = np.load('blue_median_frame.dat',allow_pickle=True)
    medians_frame_g = np.load('green_median_frame.dat',allow_pickle=True)
    medians_frame_r = np.load('red_median_frame.dat',allow_pickle=True)
    medians_frame_h = np.load('hue_median_frame.dat',allow_pickle=True)
    medians_frame_s = np.load('saturation_median_frame.dat',allow_pickle=True)
    medians_frame_v = np.load('value_median_frame.dat',allow_pickle=True)

    s_diff_from_median_list, v_diff_from_median_list, b_diff_from_median_list = [], [], []
    mask_s_list, mask_v_list, blue_mask_list, mask_or_list, mask_weighted_list = [], [], [], [], []
    original_with_or_mask_results, original_with_or_mask_and_blue_results = [], []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        curr_hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
        curr_h, curr_s, curr_v = cv2.split(curr_hsv)
        curr_b, curr_g, curr_r = cv2.split(curr)
        diff_s = np.abs(medians_frame_s - curr_s).astype(np.uint8)
        s_diff_from_median_list.append(diff_s)
        diff_v = np.abs(medians_frame_v - curr_v).astype(np.uint8)
        v_diff_from_median_list.append(diff_v)
        diff_b = np.abs(medians_frame_b - curr_b).astype(np.uint8)
        b_diff_from_median_list.append(diff_b)
        mask_s = (diff_s > np.mean(diff_s) * 5)
        mask_v = (diff_v > np.mean(diff_v) * 5)
        mask_or = (mask_s | mask_v).astype(np.uint8)

        weighted_mask = scale_matrix_0_to_255(
            (0.5 * (diff_s - np.mean(diff_s) * 5) + 0.5 * (diff_v - np.mean(diff_v) * 7) > 0).astype(np.uint8))

        kernel = np.ones((7, 7), np.uint8)
        dilation = cv2.dilate(mask_or, kernel, iterations=1)

        frame_after_or_flt = apply_mask_on_color_frame(curr, dilation)
        original_with_or_mask_results.append(frame_after_or_flt)

        # Filtering blue colors
        blue_mask = (curr_b < 140).astype(np.uint8)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=2).astype(np.uint8)
        frame_after_or_and_blue_flt = apply_mask_on_color_frame(frame_after_or_flt, blue_mask)
        original_with_or_mask_and_blue_results.append(frame_after_or_and_blue_flt)

        # mask_s = scale_matrix_0_to_255(mask_s)
        # mask_s_list.append(mask_s)
        # mask_v = scale_matrix_0_to_255(mask_v)
        # mask_v_list.append(mask_v)
        # blue_mask = scale_matrix_0_to_255(blue_mask)
        # blue_mask_list.append(blue_mask)
        # dilation = scale_matrix_0_to_255(dilation)
        # mask_or_list.append(dilation)
        # mask_weighted_list.append(weighted_mask)

    write_video('original_with_or_mask_and_blue.avi', frames=original_with_or_mask_and_blue_results, fps=fps, out_size=(w, h),
                is_color=True)

    release_video_files(cap, out)
