import cv2
import numpy as np

from utils import apply_mask_on_color_frame, check_in_dict, scale_matrix_0_to_255, SHOES_HEIGHT


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


    rectangle_shoes_mask_stacked = np.copy(fat_shoes_mask[SHOES_HEIGHT:, :])
    rectangle_shoes_mask_stacked = rectangle_shoes_mask_stacked.reshape(((h-SHOES_HEIGHT)*w))
    rectangle_shoes_mask_stacked_indices = np.where(rectangle_shoes_mask_stacked == 1)

    rectangle_shoes_original_frame_stacked = original_frame[SHOES_HEIGHT:,:].reshape(((h-SHOES_HEIGHT)*w, 3))
    only_shoes_colors = rectangle_shoes_original_frame_stacked[rectangle_shoes_mask_stacked_indices]
    shoes_only_foreground_probabilities = np.fromiter(map(lambda elem: check_in_dict(foreground_memory, elem, foreground_pdf),
                                               map(tuple, only_shoes_colors)), dtype=float)

    # shoes_onlyforeground_probabilities = foreground_probabilities.reshape((h, w))
    shoes_only_background_probabilities = np.fromiter(map(lambda elem: check_in_dict(background_memory, elem, background_pdf),
                                               map(tuple, only_shoes_colors)), dtype=float)
    # background_probabilities = background_probabilities.reshape((h, w))

    shoes_only_probs_mask_stacked = shoes_only_foreground_probabilities > shoes_only_background_probabilities
    rectangle_shoes_mask_stacked[rectangle_shoes_mask_stacked_indices] = shoes_only_probs_mask_stacked
    rectangle_shoes_mask = rectangle_shoes_mask_stacked.reshape(h-SHOES_HEIGHT,w)

    shoes_probs_mask = np.copy(fat_shoes_mask)
    shoes_probs_mask[SHOES_HEIGHT:, :] = rectangle_shoes_mask

    cv2.imwrite(f'restored_shoes_frame_{frame_index}.png', scale_matrix_0_to_255(shoes_probs_mask))
    cv2.imwrite(f'restored_shoes_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, shoes_probs_mask))

    diff_green = abs(original_frame_green-medians_frame_g)
    mask_green = (diff_green > np.mean(diff_green)*3).astype(np.uint8)
    rectangle_shoes_mask_green = mask_green[SHOES_HEIGHT:,:]
    full_shoes_mask_green = np.copy(fat_shoes_mask)
    full_shoes_mask_green[SHOES_HEIGHT:,:] = rectangle_shoes_mask_green
    full_shoes_mask_green = cv2.erode(full_shoes_mask_green, np.ones((4, 1), np.uint8), iterations=1)
    full_shoes_mask_green = cv2.erode(full_shoes_mask_green, np.ones((1, 4), np.uint8), iterations=1)
    full_shoes_mask_green = cv2.morphologyEx(full_shoes_mask_green, cv2.MORPH_CLOSE,
                                                        np.ones((10, 10), np.uint8),iterations=3)
    full_shoes_mask_green = cv2.morphologyEx(full_shoes_mask_green, cv2.MORPH_OPEN,
                                                        np.ones((2, 2), np.uint8),iterations=3)
    cv2.imwrite(f'restored_shoes_only_green_frame_{frame_index}.png', scale_matrix_0_to_255(full_shoes_mask_green))
    cv2.imwrite(f'restored_shoes_only_green_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, full_shoes_mask_green))

    shoes_probs_and_green_mask = np.maximum(shoes_probs_mask, full_shoes_mask_green)
    cv2.imwrite(f'restored_shoes_probs_and_green_frame_{frame_index}.png', scale_matrix_0_to_255(full_shoes_mask_green))
    cv2.imwrite(f'restored_shoes_probs_and_green_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, full_shoes_mask_green))

    return shoes_probs_and_green_mask
    #
    # shoes_only_probs_mask = shoes_only_probs_mask.reshape()
    # probs_mask = probs_mask.astype(np.uint8) * 255

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
