import cv2
import numpy as np

from constants import (
    BW_MEDIUM,
    NUM_SAMPLES_FOR_KDE,
    LEGS_HEIGHT,
    SHOULDERS_HEIGHT,
    OVERHEAD_HEIGHT,
    SHOES_HEIGHT
)

from utils import (
    apply_mask_on_color_frame,
    check_in_dict,
    scale_matrix_0_to_255,
    choose_indices_for_background,
    choose_indices_for_foreground,
    estimate_pdf
)


def fine_tune_contour_mask(original_frame, contour_mask,background_pdf, background_pdf_memoization, foreground_pdf,
                           foreground_pdf_memoization):
    h, w = contour_mask.shape
    dilated_contour_mask = cv2.dilate(contour_mask, np.ones((5, 5), np.uint8), iterations=3)

    row_stacked_original_frame = original_frame.reshape((h * w), 3)
    dilated_contour_mask_frame_stacked = dilated_contour_mask.reshape((h * w))
    contour_indices = np.where(dilated_contour_mask_frame_stacked == 1)
    foreground_probabilities = np.fromiter(
        map(lambda elem: check_in_dict(foreground_pdf_memoization, elem, foreground_pdf),
            map(tuple, row_stacked_original_frame[contour_indices])), dtype=float)
    background_probabilities = np.fromiter(
        map(lambda elem: check_in_dict(background_pdf_memoization, elem, background_pdf),
            map(tuple, row_stacked_original_frame[contour_indices])), dtype=float)

    contour_fine_tuning_mask = foreground_probabilities > background_probabilities
    dilated_contour_mask_frame_stacked[contour_indices] = contour_fine_tuning_mask
    dilated_contour_mask_frame_tuned = dilated_contour_mask_frame_stacked.reshape((h, w))
    dilated_contour_mask_frame_tuned = cv2.morphologyEx(dilated_contour_mask_frame_tuned, cv2.MORPH_CLOSE,
                                                        np.ones((3, 3), np.uint8), iterations=2)
    dilated_contour_mask_frame_tuned = cv2.morphologyEx(dilated_contour_mask_frame_tuned, cv2.MORPH_CLOSE,
                                                        np.ones((5, 1), np.uint8))
    return dilated_contour_mask_frame_tuned


def restore_shoes(original_frame, contour_mask, shoes_specialist_pdf):
    fat_shoes_mask = np.copy(contour_mask)

    fat_shoes_mask[:LEGS_HEIGHT, :] = 0
    fat_shoes_mask[LEGS_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[LEGS_HEIGHT:, :], np.ones((5, 5), np.uint8),
                                                 iterations=1)
    fat_shoes_mask[LEGS_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[LEGS_HEIGHT:, :], np.ones((15, 1), np.uint8),
                                                 iterations=3)

    shoes_indices_rectangle_x_axis = np.where(fat_shoes_mask == 1)[1]
    left_shoe_index, right_shoe_index = np.min(shoes_indices_rectangle_x_axis), np.max(shoes_indices_rectangle_x_axis)
    right_shoe_index += 30
    small_mask_around_shoes = fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1]
    
    '''Create background pdf around shoes'''
    omega_b_around_shoes_indices = choose_indices_for_background(small_mask_around_shoes, 300)
    background_around_shoes_pdf = estimate_pdf(
        original_frame=original_frame[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1],
        indices=omega_b_around_shoes_indices, bw_method=BW_MEDIUM)
    
    ''' Work with small piece of the frame '''
    small_original_frame_stacked = original_frame[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1].reshape((-1, 3))
    small_shoes_foreground_probabilities = shoes_specialist_pdf(small_original_frame_stacked)
    small_shoes_background_probabilities = background_around_shoes_pdf(small_original_frame_stacked)

    '''Build mask for shoes'''
    small_shoes_classifier_mask = small_shoes_foreground_probabilities / (small_shoes_background_probabilities + small_shoes_foreground_probabilities)
    small_shoes_classifier_mask = (small_shoes_classifier_mask > 0.55).astype(np.uint8)
    small_shoes_classifier_mask = small_shoes_classifier_mask.reshape((-1, right_shoe_index - left_shoe_index + 1))
    
    '''Update input mask with the new shoes'''
    shoes_mask = np.copy(contour_mask)
    shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = np.maximum(small_shoes_classifier_mask,
                                                                                    contour_mask[LEGS_HEIGHT:,
                                                                                    left_shoe_index:right_shoe_index + 1])

    shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = cv2.morphologyEx(
        shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1], cv2.MORPH_CLOSE,
        np.ones((15, 1), np.uint8), iterations=1)
    shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = cv2.morphologyEx(
        shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1], cv2.MORPH_CLOSE,
        np.ones((8, 8), np.uint8), iterations=1)

    _, contours, _ = cv2.findContours(shoes_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    one_foot_mask = np.zeros(shoes_mask.shape)
    cv2.fillPoly(one_foot_mask, pts=[contours[0]], color=1)

    if len(contours) > 1:
        second_foot_mask = np.zeros(shoes_mask.shape)
        cv2.fillPoly(second_foot_mask, pts=[contours[1]], color=1)
        person_one_foot_mask = one_foot_mask * shoes_mask
        person_second_foot_mask = second_foot_mask * shoes_mask
        person_mask = np.maximum(person_one_foot_mask, person_second_foot_mask)
    else:
        person_mask = one_foot_mask * shoes_mask

    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, np.ones((15, 1), np.uint8), iterations=1)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE,np.ones((10, 10), np.uint8), iterations=1)

    return person_mask


def build_shoes_pdf(original_frame, mask):
    _, original_frame_green, _ = cv2.split(original_frame)
    fat_shoes_mask = np.copy(mask)
    fat_shoes_mask[:LEGS_HEIGHT, :] = 0
    fat_shoes_mask[LEGS_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[LEGS_HEIGHT:, :], np.ones((5, 5), np.uint8),
                                                 iterations=1)
    fat_shoes_mask[LEGS_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[LEGS_HEIGHT:, :], np.ones((15, 1), np.uint8),
                                                 iterations=3)
    fat_shoes_mask[:SHOES_HEIGHT, :] = 0
    fat_shoes_mask[970:, :] = 0
    for x in range(255, 355):
        for y in range(SHOES_HEIGHT, 970):
            fat_shoes_mask[y, x] = 0 if y < 0.4 * x + 773 else 1
    fat_shoes_mask[954:, 255:] = 0

    omega_f_around_shoes_indices = choose_indices_for_foreground(fat_shoes_mask, 300)
    foreground_around_shoes_pdf = estimate_pdf(
        original_frame=original_frame,
        indices=omega_f_around_shoes_indices, bw_method=BW_MEDIUM)

    return foreground_around_shoes_pdf


def build_shoulders_face_pdf(mask, image, bw_method):
    small_mask = mask[:SHOULDERS_HEIGHT, :]
    small_image = image[:SHOULDERS_HEIGHT, :]
    small_mask = cv2.erode(small_mask, np.ones((2, 2)), iterations=1)
    omega_f_indices = choose_indices_for_foreground(small_mask, NUM_SAMPLES_FOR_KDE)
    foreground_pdf = estimate_pdf(original_frame=small_image, indices=omega_f_indices, bw_method=bw_method)
    return foreground_pdf


def remove_signs(original_frame, mask, shoulders_and_face_narrow_pdf, shoulders_and_face_wide_pdf,
                 shoulders_and_face_pdf_narrow_memoization, shoulders_and_face_pdf_wide_memoization):
    person_mask = np.copy(mask)
    face_indices_rectangle_x_axis = np.where(person_mask == 1)[1]
    left_index, right_index = np.min(face_indices_rectangle_x_axis), np.max(face_indices_rectangle_x_axis)
    small_image = original_frame[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1]
    small_image_row_stacked = small_image.reshape(-1, 3)

    ''' REMOVE BACKGROUND WITH NARROW BAND FACE PDF'''
    foreground_face_narrow_probabilities = np.fromiter(
        map(lambda elem: check_in_dict(shoulders_and_face_pdf_narrow_memoization, elem, shoulders_and_face_narrow_pdf),
            map(tuple, small_image_row_stacked)), dtype=float)
    small_mask_only_narrow_foreground_row_stacked = (
            foreground_face_narrow_probabilities > 0.1 * np.mean(foreground_face_narrow_probabilities)).astype(np.uint8)

    small_mask_foreground_face_narrow = small_mask_only_narrow_foreground_row_stacked.reshape(
        (SHOULDERS_HEIGHT - OVERHEAD_HEIGHT, right_index - left_index + 1))

    ''' COMPARE FOREGROUND WITH BACKGROUND '''
    foreground_face_wide_probabilities = np.fromiter(
        map(lambda elem: check_in_dict(shoulders_and_face_pdf_wide_memoization, elem, shoulders_and_face_wide_pdf),
            map(tuple, small_image_row_stacked)), dtype=float)

    omega_b_around_face_indices = choose_indices_for_background(small_mask_foreground_face_narrow, 300)
    background_around_face_pdf = estimate_pdf(
        original_frame=original_frame[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1],
        indices=omega_b_around_face_indices, bw_method=BW_MEDIUM)
    background_around_face_probabilites = background_around_face_pdf(small_image_row_stacked)

    small_mask_probs_fg_bigger_bg_stacked = (foreground_face_wide_probabilities > background_around_face_probabilites).astype(np.uint8)

    small_mask_fg_bigger_bg_stacked = small_mask_probs_fg_bigger_bg_stacked * small_mask_only_narrow_foreground_row_stacked  # TODO - MAYBE MAXIMUM BETTER OPTION

    small_mask_fg_bigger_bg = small_mask_fg_bigger_bg_stacked.reshape((SHOULDERS_HEIGHT - OVERHEAD_HEIGHT, right_index - left_index + 1))
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = small_mask_fg_bigger_bg

    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.morphologyEx(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8), iterations=2)
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.ones((5, 1)), iterations=1)
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.ones((1, 5)), iterations=1)
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.identity(3, dtype=np.uint8),
        iterations=1)

    inverse_identity = np.zeros((3, 3), dtype=np.uint8)
    inverse_identity[2, 0] = 1
    inverse_identity[1, 1] = 1
    inverse_identity[0, 2] = 1
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], inverse_identity, iterations=1)

    _, contours, _ = cv2.findContours(person_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    contour_mask = np.zeros(person_mask.shape)
    cv2.fillPoly(contour_mask, pts=[contours[0]], color=1)
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = contour_mask[
                                                                                OVERHEAD_HEIGHT:SHOULDERS_HEIGHT,
                                                                                left_index:right_index + 1]

    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.dilate(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.ones((3, 3), dtype=np.uint8),
        iterations=1)

    return person_mask
