SHOES_HEIGHT = 855
import cv2
import numpy as np

from utils import apply_mask_on_color_frame, check_in_dict, scale_matrix_0_to_255


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
                            medians_frame_g):
    h, w = contour_mask.shape
    _,original_frame_green,_ = cv2.split(original_frame)
    fat_shoes_mask = np.copy(contour_mask)
    fat_shoes_mask[:SHOES_HEIGHT,:] = 0
    fat_shoes_mask[SHOES_HEIGHT:,:] = cv2.dilate(fat_shoes_mask[SHOES_HEIGHT:,:], np.ones((5, 5), np.uint8), iterations=3)
    diff_green = abs(original_frame_green-medians_frame_g)
    mask_green = (diff_green > np.mean(diff_green)).astype(np.uint8)
    fat_shoes_green_filtered = mask_green * fat_shoes_mask
    fat_shoes_green_filtered[:SHOES_HEIGHT,:] = contour_mask[:SHOES_HEIGHT,:]

    # dilated_contour_mask = cv2.dilate(contour_mask, np.ones((5, 5), np.uint8), iterations=3)
    cv2.imwrite(f'restored_shoes_frame_{frame_index}.png', scale_matrix_0_to_255(fat_shoes_green_filtered))
    cv2.imwrite(f'restored_shoes_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, fat_shoes_green_filtered))


    #
    # contour_fine_tuning_mask = foreground_probabilities + 0 > background_probabilities
    # dilated_contour_mask_frame_stacked[contour_indices] = contour_fine_tuning_mask
    # dilated_contour_mask_frame_tuned = dilated_contour_mask_frame_stacked.reshape((h, w))
    # dilated_contour_mask_frame_tuned = cv2.morphologyEx(dilated_contour_mask_frame_tuned, cv2.MORPH_CLOSE,
    #                                                     np.ones((3, 3), np.uint8), iterations=2)
    # dilated_contour_mask_frame_tuned = cv2.morphologyEx(dilated_contour_mask_frame_tuned, cv2.MORPH_CLOSE,
    #                                                     np.ones((5, 1), np.uint8))
    #
    # cv2.imwrite(f'test_color_fine_tune_no_eps_closing_{frame_index}.png',
    #             apply_mask_on_color_frame(original_frame, dilated_contour_mask_frame_tuned))
