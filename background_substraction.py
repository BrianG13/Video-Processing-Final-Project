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
    choose_indices_for_background,
    check_in_dict,
    SHOES_HEIGHT,
    FRAME_INDEX_FOR_FACE_TUNING
)
from kernel_estimation import (
    estimate_pdf
)

from fine_tune_background_substraction import (
    fine_tune_contour_mask,
    restore_shoes,
    build_shoulders_face_pdf,
    remove_signs
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

    medians_frame_b = np.load('blue_median_frame.dat', allow_pickle=True)
    medians_frame_g = np.load('green_median_frame.dat', allow_pickle=True)
    medians_frame_r = np.load('red_median_frame.dat', allow_pickle=True)
    medians_frame_h = np.load('hue_median_frame.dat', allow_pickle=True)
    medians_frame_s = np.load('saturation_median_frame.dat', allow_pickle=True)
    medians_frame_v = np.load('value_median_frame.dat', allow_pickle=True)

    s_diff_from_median_list, v_diff_from_median_list, b_diff_from_median_list = [], [], []
    mask_s_list, mask_v_list, blue_mask_list, mask_or_list, mask_weighted_list = [], [], [], [], []
    original_with_or_mask_results, original_with_or_mask_and_blue_results = [], []
    probs_mask_after_closing_list = []
    probs_mask_eroison_list = []
    contour_color_list = []
    removed_signs_color_frame_list, removed_signs_mask_list = [], []
    fine_tuned_with_shoes_color_list, fine_tuned_with_shoes_mask_list = [], []
    '''Build KDE from specified frame filtered by s,v channels'''
    mask_for_building_kde = frame_92(frames_bgr[92])
    omega_f_indices = choose_indices_for_foreground(mask_for_building_kde, 200)
    omega_b_indices = choose_indices_for_background(mask_for_building_kde, 200)
    foreground_pdf = estimate_pdf(original_frame=frames_bgr[92], indices=omega_f_indices, bw_method=2)
    background_pdf = estimate_pdf(original_frame=frames_bgr[92], indices=omega_b_indices, bw_method=2)
    foreground_memory = dict()
    background_memory = dict()

    ''''''

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

        row_stacked_original_frame = curr.reshape((h * w), 3)
        foreground_probabilities = np.fromiter(map(lambda elem: check_in_dict(foreground_memory, elem, foreground_pdf),
                                                   map(tuple, row_stacked_original_frame)), dtype=float)
        foreground_probabilities = foreground_probabilities.reshape((h, w))
        background_probabilities = np.fromiter(map(lambda elem: check_in_dict(background_memory, elem, background_pdf),
                                                   map(tuple, row_stacked_original_frame)), dtype=float)
        background_probabilities = background_probabilities.reshape((h, w))

        probs_mask = foreground_probabilities > background_probabilities
        probs_mask = probs_mask.astype(np.uint8) * 255

        probs_mask_eroison = cv2.erode(probs_mask, np.ones((3, 1), np.uint8), iterations=2)
        probs_mask_eroison = cv2.erode(probs_mask_eroison, np.ones((1, 3), np.uint8), iterations=1)

        cv2.imwrite(f'probs_mask_eroison_frame_{i}.png', probs_mask_eroison)
        probs_mask_eroison_list.append(probs_mask_eroison)

        closing = cv2.morphologyEx(probs_mask_eroison, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        closing = cv2.morphologyEx(probs_mask_eroison, cv2.MORPH_CLOSE, np.ones((10, 1), np.uint8))
        probs_mask_after_closing_list.append(closing)
        cv2.imwrite(f'probs_mask_frame_closing_{i}.png', closing)

        img, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # cv2.drawContours(curr, contours, 0, (0, 255, 0), 3)
        # cv2.imwrite(f'contours_probs_{i}.png', curr)
        contour_mask = np.zeros((h, w))  # create a single channel 200x200 pixel black image
        cv2.fillPoly(contour_mask, pts=[contours[0]], color=1)
        cv2.imwrite(f'filled_contour_img_{i}.png', scale_matrix_0_to_255(contour_mask))
        contour_color_image = apply_mask_on_color_frame(curr, contour_mask)
        contour_color_list.append(contour_color_image)
        cv2.imwrite(f'contours_color_img_{i}.png', contour_color_image)

        mask_fine_tuned_after_contours = fine_tune_contour_mask(frame_index=i,
                                                                contour_mask=contour_mask,
                                                                original_frame=curr,
                                                                background_pdf=background_pdf,
                                                                background_memory=background_memory,
                                                                foreground_pdf=foreground_pdf,
                                                                foreground_memory=foreground_memory)
        mask_after_shoes_fix = restore_shoes(frame_index=i,
                                             contour_mask=mask_fine_tuned_after_contours,
                                             original_frame=curr,
                                             background_pdf=background_pdf,
                                             background_memory=background_memory,
                                             foreground_pdf=foreground_pdf,
                                             foreground_memory=foreground_memory,
                                             medians_frame_g=medians_frame_g,
                                             mask_to_classify_shoes_from_ground=mask_for_building_kde)

        mask_fine_tuned_with_shoes = np.copy(mask_fine_tuned_after_contours)
        mask_fine_tuned_with_shoes[SHOES_HEIGHT:, :] = mask_after_shoes_fix[SHOES_HEIGHT:, :]
        mask_fine_tuned_with_shoes = cv2.erode(mask_fine_tuned_with_shoes, np.ones((4, 1), np.uint8), iterations=4)
        mask_fine_tuned_with_shoes = cv2.dilate(mask_fine_tuned_with_shoes, np.ones((4, 1), np.uint8), iterations=4)
        fine_tuned_with_shoes_color = apply_mask_on_color_frame(curr, mask_fine_tuned_with_shoes)
        fine_tuned_with_shoes_color_list.append(fine_tuned_with_shoes_color)
        fine_tuned_with_shoes_mask_list.append(mask_fine_tuned_with_shoes)
        cv2.imwrite(f'fine_tune_contours_and_shoes_{i}.png',
                    apply_mask_on_color_frame(curr, mask_fine_tuned_with_shoes))
        cv2.imwrite(f'fine_tune_contours_and_shoes_mask_{i}.png',
                    scale_matrix_0_to_255(mask_fine_tuned_with_shoes))


    fine_tuned_with_shoes_mask_list = load_masks_from_middle()  # TODO- DELETE THIS HACK!
    shoulders_face_narrow_pdf = build_shoulders_face_pdf(fine_tuned_with_shoes_mask_list[FRAME_INDEX_FOR_FACE_TUNING],
                                                         frames_bgr[FRAME_INDEX_FOR_FACE_TUNING],
                                                         bw_method=0.3)
    shoulders_face_wide_pdf = build_shoulders_face_pdf(fine_tuned_with_shoes_mask_list[FRAME_INDEX_FOR_FACE_TUNING],
                                                       frames_bgr[FRAME_INDEX_FOR_FACE_TUNING],
                                                       bw_method=1)
    shoulders_and_face_pdf_narrow_memory = dict()
    shoulders_and_face_pdf_wide_memory = dict()
    for frame_index in range(n_frames):
        print(frame_index)
        remove_signs_mask = remove_signs(frame_index=frame_index,
                                         original_frame=frames_bgr[frame_index],
                                         mask=fine_tuned_with_shoes_mask_list[frame_index],
                                         shoulders_and_face_narrow_pdf=shoulders_face_narrow_pdf,
                                         shoulders_and_face_wide_pdf=shoulders_face_wide_pdf,
                                         shoulders_and_face_pdf_narrow_memory=shoulders_and_face_pdf_narrow_memory,
                                         shoulders_and_face_pdf_wide_memory=shoulders_and_face_pdf_wide_memory
                                         )
        removed_signs_color_frame = apply_mask_on_color_frame(frames_bgr[frame_index], remove_signs_mask)
        removed_signs_color_frame_list.append(removed_signs_color_frame)
        removed_signs_mask_list.append(scale_matrix_0_to_255(remove_signs_mask))

    write_video('background_substraction_mask.avi', frames=removed_signs_mask_list, fps=fps, out_size=(w, h),
                is_color=False)
    write_video('background_substraction.avi', frames=removed_signs_color_frame_list, fps=fps, out_size=(w, h),
                is_color=True)
    write_video('probs_mask_after_erosion_before_closing.avi', frames=probs_mask_eroison_list, fps=fps, out_size=(w, h),
                is_color=False)
    write_video('probs_mask_after_closing.avi', frames=probs_mask_after_closing_list, fps=fps, out_size=(w, h),
                is_color=False)
    write_video('original_only_contour.avi', frames=contour_color_list, fps=fps, out_size=(w, h), is_color=True)
    release_video_files(cap, out)


def frame_92(curr):
    medians_frame_s = np.load('saturation_median_frame.dat', allow_pickle=True)
    medians_frame_v = np.load('value_median_frame.dat', allow_pickle=True)
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

    final_mask = dilation * blue_mask
    final_mask = cv2.erode(final_mask, np.ones((5, 5), np.uint8), iterations=4)
    # final_mask = cv2.erode(final_mask, np.ones((5, 1), np.uint8), iterations=3)

    cv2.imwrite('frame92_msk.png', scale_matrix_0_to_255(final_mask))
    return final_mask


def load_masks_from_middle():
    frames = []
    for i in range(205):
        frame = cv2.imread(f'fine_tune_contours_and_shoes_mask_{i}.png', cv2.IMREAD_GRAYSCALE)
        frame = frame / 255
        frames.append(frame)
    return frames
