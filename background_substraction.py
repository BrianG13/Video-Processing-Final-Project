import cv2
import numpy as np

from utils import (
    get_video_files,
    release_video_files,
    write_video,
    load_entire_video,
    scale_matrix_0_to_255,
    apply_mask_on_color_frame,
    choose_indices_for_foreground,
    choose_indices_for_background
)
from kernel_estimation import (
    estimate_pdf
)

import time

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
    probs_mask_list = []

    mask = frame_92(frames_bgr[92])
    # cv2.imshow('sas', scale_matrix_0_to_255(mask))
    # cv2.waitKey(0)
    omega_f_indices = choose_indices_for_foreground(mask, 200)
    # image = np.copy(frames_bgr[92])
    # for index in range(omega_f_indices.shape[0]):
    #     image = cv2.circle(image, (omega_f_indices[index][1], omega_f_indices[index][0]), 5, (0, 255, 0), 2)
    # cv2.imshow('sas', image)
    # cv2.waitKey(0)

    omega_b_indices = choose_indices_for_background(mask, 200)
    # image = np.copy(frames_bgr[92])
    # for index in range(omega_b_indices.shape[0]):
    #     image = cv2.circle(image, (omega_b_indices[index][1], omega_b_indices[index][0]), 5, (0, 255, 0), 2)
    # cv2.imshow('sas', image)
    # cv2.waitKey(0)

    foreground_pdf = estimate_pdf(original_frame=frames_bgr[92], indices=omega_f_indices)
    background_pdf = estimate_pdf(original_frame=frames_bgr[92], indices=omega_b_indices)
    foreground_memory = dict()
    background_memory = dict()
    for i in range(n_frames):
        print("Frame: " + str(i) + "/" + str(n_frames))
        success, curr = cap.read()
        if not success:
            break
        '''COMMENTING THIS, LOADING FRAME 92, SO ALL THIS CALCULCATIONS OVER TIME ARE NOT NECESSARY'''
        # curr_hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
        # curr_h, curr_s, curr_v = cv2.split(curr_hsv)
        # curr_b, curr_g, curr_r = cv2.split(curr)
        # diff_s = np.abs(medians_frame_s - curr_s).astype(np.uint8)
        # s_diff_from_median_list.append(diff_s)
        # diff_v = np.abs(medians_frame_v - curr_v).astype(np.uint8)
        # v_diff_from_median_list.append(diff_v)
        # diff_b = np.abs(medians_frame_b - curr_b).astype(np.uint8)
        # b_diff_from_median_list.append(diff_b)
        # mask_s = (diff_s > np.mean(diff_s) * 5)
        # mask_v = (diff_v > np.mean(diff_v) * 5)
        # mask_or = (mask_s | mask_v).astype(np.uint8)

        # weighted_mask = scale_matrix_0_to_255(
        #     (0.5 * (diff_s - np.mean(diff_s) * 5) + 0.5 * (diff_v - np.mean(diff_v) * 7) > 0).astype(np.uint8))

        # kernel = np.ones((1, 1), np.uint8)
        # dilation = cv2.dilate(mask_or, kernel, iterations=1)
        #
        # frame_after_or_flt = apply_mask_on_color_frame(curr, dilation)
        # original_with_or_mask_results.append(frame_after_or_flt)
        #
        # # Filtering blue colors
        # blue_mask = (curr_b < 140).astype(np.uint8)
        # blue_kernel = np.ones((3, 3), np.uint8)
        # blue_mask = cv2.erode(blue_mask, blue_kernel, iterations=2).astype(np.uint8)
        # frame_after_or_and_blue_flt = apply_mask_on_color_frame(frame_after_or_flt, blue_mask)
        # original_with_or_mask_and_blue_results.append(frame_after_or_and_blue_flt)
        '''END OF BIG COMMENT'''



        # print('FOREGROUND PDF!')
        # print(f'black: {foreground_pdf(np.asarray([0,0,0]).T)}')
        # print(f'red: {foreground_pdf(np.asarray([100,100,255]).T)}')
        # print(f'blue: {foreground_pdf(np.asarray([255,0,0]).T)}')
        # print(f'green: {foreground_pdf(np.asarray([0,255,0]).T)}')
        # print(f'white: {foreground_pdf(np.asarray([255,255,255]).T)}')
        # print(f'shirt: {foreground_pdf(np.asarray([54,41,71]).T)}')
        # print(f'shoe: {foreground_pdf(np.asarray([90,90,90]).T)}')
        # print('BACKGROUND PDF!')
        # print(f'black: {background_pdf(np.asarray([0,0,0]).T)}')
        # print(f'red: {background_pdf(np.asarray([100,100,255]).T)}')
        # print(f'blue: {background_pdf(np.asarray([255,0,0]).T)}')
        # print(f'green: {background_pdf(np.asarray([0,255,0]).T)}')
        # print(f'white: {background_pdf(np.asarray([255,255,255]).T)}')
        # print(f'shirt: {background_pdf(np.asarray([54,41,71]).T)}')
        # print(f'shoe: {background_pdf(np.asarray([90,90,90]).T)}')

        row_stacked_original_frame = curr.reshape((h*w),3)
        now = time.time()
        foreground_probabilities = np.fromiter(map(lambda elem: check_in_dict(foreground_memory,elem,foreground_pdf),
                                         map(tuple, row_stacked_original_frame)),dtype=float)
        foreground_probabilities = foreground_probabilities.reshape((h,w))
        background_probabilities = np.fromiter(map(lambda elem: check_in_dict(background_memory,elem,background_pdf),
                                         map(tuple, row_stacked_original_frame)),dtype=float)
        background_probabilities = background_probabilities.reshape((h,w))
        print(f'background mem size: {len(background_memory)}')
        print(f'foreground mem size: {len(foreground_memory)}')
        print(f'time for probs : {time.time()-now}')

        probs_mask = foreground_probabilities > background_probabilities
        probs_mask = probs_mask.astype(np.uint8) * 255
        probs_mask_list.append(probs_mask)
        cv2.imwrite(f'probs_mask_frame_{i}.png',probs_mask)

        probs_mask_eroison = cv2.erode(probs_mask, np.ones((3, 1), np.uint8), iterations=1)
        probs_mask_eroison = cv2.erode(probs_mask_eroison, np.ones((1, 3), np.uint8), iterations=1)

        # probs_mask_eroison = cv2.erode(probs_mask_eroison, np.ones((3, 1), np.uint8), iterations=1)
        # probs_mask_eroison = cv2.erode(probs_mask_eroison, np.ones((1, 4), np.uint8), iterations=1)
        closing = cv2.morphologyEx(probs_mask_eroison, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        img, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)
        cv2.drawContours(curr, contours, 0, (0, 255, 0), 3)
        cv2.imwrite(f'countours_probs_{i}.png',curr)

        cv2.imwrite(f'probs_mask_frame_eroded_{i}.png',closing)




        # mask_s = scale_matrix_0_to_255(mask_s)
        # mask_s_list.append(mask_s)
        # mask_v = scale_matrix_0_to_255(mask_v)
        # mask_v_list.append(mask_v)
        # blue_mask = scale_matrix_0_to_255(blue_mask)
        # blue_mask_list.append(blue_mask)
        # dilation = scale_matrix_0_to_255(dilation)
        # mask_or_list.append(dilation)
        # mask_weighted_list.append(weighted_mask)

    # write_video('original_with_or_mask_and_blue.avi', frames=original_with_or_mask_and_blue_results, fps=fps, out_size=(w, h),
    #             is_color=True)
    write_video('probs_mask.avi', frames=probs_mask_list, fps=fps, out_size=(w, h),is_color=False)
    release_video_files(cap, out)


def frame_92(curr):
    medians_frame_s = np.load('saturation_median_frame.dat',allow_pickle=True)
    medians_frame_v = np.load('value_median_frame.dat',allow_pickle=True)
    curr_hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    curr_h, curr_s, curr_v = cv2.split(curr_hsv)
    curr_b, curr_g, curr_r = cv2.split(curr)
    diff_s = np.abs(medians_frame_s - curr_s).astype(np.uint8)
    diff_v = np.abs(medians_frame_v - curr_v).astype(np.uint8)
    mask_s = (diff_s > np.mean(diff_s) * 5)
    mask_v = (diff_v > np.mean(diff_v) * 5)
    mask_or = (mask_s | mask_v).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(mask_or, kernel, iterations=1)
    # Filtering blue colors
    blue_mask = (curr_b < 140).astype(np.uint8)
    blue_kernel = np.ones((3, 3), np.uint8)
    blue_mask = cv2.erode(blue_mask, blue_kernel, iterations=2).astype(np.uint8)

    final_mask = dilation*blue_mask
    final_mask = cv2.erode(final_mask, np.ones((5, 5), np.uint8), iterations=4)
    # final_mask = cv2.erode(final_mask, np.ones((5, 1), np.uint8), iterations=3)

    cv2.imwrite('frame92_msk.png',scale_matrix_0_to_255(final_mask))
    return final_mask

def check_in_dict(dict,element,function):
    if element in dict:
        return dict[element]
    else:
        dict[element] = function(np.asarray(element))[0]
        return dict[element]
