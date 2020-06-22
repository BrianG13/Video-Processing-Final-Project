import cv2
import numpy as np

from utils import (
    apply_mask_on_color_frame,
    check_in_dict,
    scale_matrix_0_to_255,
    SHOES_HEIGHT,
    choose_indices_for_background,
    choose_indices_for_foreground,
)

from kernel_estimation import estimate_pdf


def fine_tune_contour_mask(frame_index, original_frame, contour_mask,
                           background_pdf, background_memory, foreground_pdf, foreground_memory):
    h, w = contour_mask.shape
    dilated_contour_mask = cv2.dilate(contour_mask, np.ones((5, 5), np.uint8), iterations=3)
    cv2.imwrite(f'dilated_contour_mask_frame_{frame_index}.png', scale_matrix_0_to_255(dilated_contour_mask))
    cv2.imwrite(f'dilated_contour_mask_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, dilated_contour_mask))

    row_stacked_original_frame = original_frame.reshape((h * w), 3)
    dilated_contour_mask_frame_stacked = dilated_contour_mask.reshape((h * w))
    contour_indices = np.where(dilated_contour_mask_frame_stacked == 1)
    foreground_probabilities = np.fromiter(map(lambda elem: check_in_dict(foreground_memory, elem, foreground_pdf),
                                               map(tuple, row_stacked_original_frame[contour_indices])), dtype=float)
    background_probabilities = np.fromiter(map(lambda elem: check_in_dict(background_memory, elem, background_pdf),
                                               map(tuple, row_stacked_original_frame[contour_indices])), dtype=float)

    contour_fine_tuning_mask = foreground_probabilities + 0 > background_probabilities
    dilated_contour_mask_frame_stacked[contour_indices] = contour_fine_tuning_mask
    dilated_contour_mask_frame_tuned = dilated_contour_mask_frame_stacked.reshape((h, w))
    dilated_contour_mask_frame_tuned = cv2.morphologyEx(dilated_contour_mask_frame_tuned, cv2.MORPH_CLOSE,
                                                        np.ones((3, 3), np.uint8), iterations=2)
    dilated_contour_mask_frame_tuned = cv2.morphologyEx(dilated_contour_mask_frame_tuned, cv2.MORPH_CLOSE,
                                                        np.ones((5, 1), np.uint8))

    cv2.imwrite(f'test_color_fine_tune_no_eps_closing_{frame_index}.png',
                apply_mask_on_color_frame(original_frame, dilated_contour_mask_frame_tuned))

    return dilated_contour_mask_frame_tuned


def restore_shoes(frame_index, original_frame, contour_mask,
                  background_pdf, background_memory, foreground_pdf, foreground_memory,
                  medians_frame_g, mask_to_classify_shoes_from_ground):
    _, original_frame_green, _ = cv2.split(original_frame)
    fat_shoes_mask = np.copy(contour_mask)
    fat_shoes_mask[:SHOES_HEIGHT, :] = 0
    fat_shoes_mask[SHOES_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[SHOES_HEIGHT:, :], np.ones((5, 5), np.uint8),
                                                  iterations=1)
    fat_shoes_mask[SHOES_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[SHOES_HEIGHT:, :], np.ones((15, 1), np.uint8),
                                                  iterations=3)

    # cv2.imshow('ss',scale_matrix_0_to_255(fat_shoes_mask))
    # cv2.waitKey(0)
    shoes_indices_rectangle_x_axis = np.where(fat_shoes_mask == 1)[1]
    left_shoe_index, right_shoe_index = np.min(shoes_indices_rectangle_x_axis), np.max(shoes_indices_rectangle_x_axis)
    right_shoe_index += 30
    small_mask_around_shoes = fat_shoes_mask[SHOES_HEIGHT:, left_shoe_index:right_shoe_index + 1]
    omega_f_around_shoes_indices = choose_indices_for_foreground(small_mask_around_shoes, 300)
    omega_b_around_shoes_indices = choose_indices_for_background(small_mask_around_shoes, 300)

    # image = np.copy(original_frame[SHOES_HEIGHT:,left_shoe_index:right_shoe_index+1])
    # for index in range(omega_f_around_shoes_indices.shape[0]):
    #     image = cv2.circle(image, (omega_f_around_shoes_indices[index][1], omega_f_around_shoes_indices[index][0]), 2, (0, 255, 0), 1)
    # ## Displaying the image
    # cv2.imshow('foreground', image)
    # # cv2.waitKey(0)
    #
    # image = np.copy(original_frame[SHOES_HEIGHT:,left_shoe_index:right_shoe_index+1])
    # for index in range(omega_b_around_shoes_indices.shape[0]):
    #     image = cv2.circle(image, (omega_b_around_shoes_indices[index][1], omega_b_around_shoes_indices[index][0]), 2, (0, 255, 0), 1)
    # ## Displaying the image
    # cv2.imshow('background', image)
    # cv2.waitKey(0)

    foreground_around_shoes_pdf = estimate_pdf(
        original_frame=original_frame[SHOES_HEIGHT:, left_shoe_index:right_shoe_index + 1],
        indices=omega_f_around_shoes_indices, bw_method=3)
    background_around_shoes_pdf = estimate_pdf(
        original_frame=original_frame[SHOES_HEIGHT:, left_shoe_index:right_shoe_index + 1],
        indices=omega_b_around_shoes_indices, bw_method=1)
    small_original_frame_stacked = original_frame[SHOES_HEIGHT:, left_shoe_index:right_shoe_index + 1].reshape((-1, 3))
    around_shoes_foreground_probabilities = foreground_around_shoes_pdf(small_original_frame_stacked)
    around_shoes_background_probabilities = background_around_shoes_pdf(small_original_frame_stacked)

    around_shoes_classifier_mask = around_shoes_foreground_probabilities > around_shoes_background_probabilities
    around_shoes_classifier_mask = around_shoes_classifier_mask.reshape((-1, right_shoe_index - left_shoe_index + 1))
    fat_shoes_mask[SHOES_HEIGHT:, left_shoe_index:right_shoe_index + 1] = around_shoes_classifier_mask

    fat_shoes_mask[SHOES_HEIGHT:, left_shoe_index:right_shoe_index + 1] = cv2.morphologyEx(
        fat_shoes_mask[SHOES_HEIGHT:, left_shoe_index:right_shoe_index + 1], cv2.MORPH_CLOSE,
        np.ones((10, 10), np.uint8), iterations=3)

    cv2.imwrite(f'classifer_shoes_frame_{frame_index}.png', scale_matrix_0_to_255(fat_shoes_mask))
    cv2.imwrite(f'classifer_shoes_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, fat_shoes_mask))

    return fat_shoes_mask


def remove_signs(frame_index, original_frame, contour_mask,
                  background_pdf, background_memory, foreground_pdf, foreground_memory,
                  medians_frame_g, mask_to_classify_shoes_from_ground):
