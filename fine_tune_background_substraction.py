import cv2
import numpy as np

from utils import (
    apply_mask_on_color_frame,
    check_in_dict,
    scale_matrix_0_to_255,
    LEGS_HEIGHT,
    SHOULDERS_HEIGHT,
    OVERHEAD_HEIGHT,
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
    cv2.imwrite(f'fine_tune_no_eps_closing_mask_{frame_index}.png',
                scale_matrix_0_to_255(dilated_contour_mask_frame_tuned))

    return dilated_contour_mask_frame_tuned


def restore_shoes(frame_index, original_frame, contour_mask,shoes_specialist_pdf,
                  background_pdf, background_memory, foreground_pdf, foreground_memory,
                  medians_frame_g, mask_to_classify_shoes_from_ground):
    _, original_frame_green, _ = cv2.split(original_frame)
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


    background_around_shoes_pdf = estimate_pdf(
        original_frame=original_frame[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1],
        indices=omega_b_around_shoes_indices, bw_method=1)

    small_original_frame_stacked = original_frame[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1].reshape((-1, 3))
    around_shoes_foreground_probabilities = shoes_specialist_pdf(small_original_frame_stacked)
    around_shoes_background_probabilities = background_around_shoes_pdf(small_original_frame_stacked)

    around_shoes_classifier_mask = around_shoes_foreground_probabilities / (around_shoes_background_probabilities + around_shoes_foreground_probabilities)
    around_shoes_classifier_mask = (around_shoes_classifier_mask > 0.55).astype(np.uint8)
    around_shoes_classifier_mask = around_shoes_classifier_mask.reshape((-1, right_shoe_index - left_shoe_index + 1))
    fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = np.maximum(around_shoes_classifier_mask,contour_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1])

    cv2.imwrite(f'classifer_shoes_before_close_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, fat_shoes_mask))

    fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = cv2.morphologyEx(
        fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1], cv2.MORPH_CLOSE,
        np.ones((15, 1), np.uint8), iterations=1)
    fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = cv2.morphologyEx(
        fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1], cv2.MORPH_CLOSE,
        np.ones((8, 8), np.uint8), iterations=1)
    cv2.imwrite(f'shoes_close_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, fat_shoes_mask))
    # fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = cv2.morphologyEx(
    #     fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1], cv2.MORPH_OPEN,
    #     np.ones((6, 1), np.uint8), iterations=1)
    # fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1] = cv2.morphologyEx(
    #     fat_shoes_mask[LEGS_HEIGHT:, left_shoe_index:right_shoe_index + 1], cv2.MORPH_OPEN,
    #     np.ones((1, 6), np.uint8), iterations=1)
    # cv2.imwrite(f'shoes_open_{frame_index}_color.png',
    #             apply_mask_on_color_frame(original_frame, fat_shoes_mask))


    #####
    img, contours, hierarchy = cv2.findContours(fat_shoes_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    one_foot_mask = np.zeros(fat_shoes_mask.shape)
    cv2.fillPoly(one_foot_mask, pts=[contours[0]], color=1)

    if len(contours)>1:
        second_foot_mask = np.zeros(fat_shoes_mask.shape)
        cv2.fillPoly(second_foot_mask, pts=[contours[1]], color=1)
        person_one_foot_mask = one_foot_mask * fat_shoes_mask
        person_second_foot_mask = second_foot_mask * fat_shoes_mask
        person_mask = np.maximum(person_one_foot_mask,person_second_foot_mask)
    else:
        person_mask = one_foot_mask * fat_shoes_mask

    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, np.ones((15, 1), np.uint8), iterations=1)

    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE,
                                    np.ones((10, 10), np.uint8), iterations=1)

    cv2.imwrite(f'filled_shoes_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))

    return person_mask


def build_shoes_pdf(frame_index,original_frame,mask,bw_method):

    _, original_frame_green, _ = cv2.split(original_frame)
    fat_shoes_mask = np.copy(mask)
    ### HACK LOAD # TODO
    fat_shoes_mask = cv2.imread('fine_tune_no_eps_closing_mask_6.png',cv2.IMREAD_GRAYSCALE)
    fat_shoes_mask = fat_shoes_mask / 255
    ## END HACK LOAD
    fat_shoes_mask[:LEGS_HEIGHT, :] = 0
    fat_shoes_mask[LEGS_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[LEGS_HEIGHT:, :], np.ones((5, 5), np.uint8),
                                                 iterations=1)
    fat_shoes_mask[LEGS_HEIGHT:, :] = cv2.dilate(fat_shoes_mask[LEGS_HEIGHT:, :], np.ones((15, 1), np.uint8),
                                                 iterations=3)
    fat_shoes_mask[:SHOES_HEIGHT, :] = 0
    fat_shoes_mask[970:, :] = 0
    for x in range(255,355):
        for y in range(SHOES_HEIGHT,970):
            fat_shoes_mask[y,x] = 0 if y < 0.4*x +773 else 1
    fat_shoes_mask[954:,255:] = 0

    omega_f_around_shoes_indices = choose_indices_for_foreground(fat_shoes_mask, 300)
    foreground_around_shoes_pdf = estimate_pdf(
        original_frame=original_frame,
        indices=omega_f_around_shoes_indices, bw_method=bw_method)

    return foreground_around_shoes_pdf

def build_shoulders_face_pdf(mask, image, bw_method):
    small_mask = mask[:SHOULDERS_HEIGHT, :]
    small_image = image[:SHOULDERS_HEIGHT, :]
    small_mask = cv2.erode(small_mask, np.ones((2, 2)), iterations=1)
    omega_f_indices = choose_indices_for_foreground(small_mask, 200)
    foreground_pdf = estimate_pdf(original_frame=small_image, indices=omega_f_indices, bw_method=bw_method)
    return foreground_pdf


def remove_signs(frame_index, original_frame, mask, shoulders_and_face_narrow_pdf, shoulders_and_face_wide_pdf,
                 shoulders_and_face_pdf_narrow_memory, shoulders_and_face_pdf_wide_memory):
    person_mask = np.copy(mask)
    face_indices_rectangle_x_axis = np.where(person_mask == 1)[1]
    left_index, right_index = np.min(face_indices_rectangle_x_axis), np.max(face_indices_rectangle_x_axis)
    small_image = original_frame[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1]
    small_image_row_stacked = small_image.reshape(-1, 3)

    ''' REMOVE BACKGROUND WITH NARROW BAND FACE PDF'''
    shoulders_face_narrow_probabilities = np.fromiter(
        map(lambda elem: check_in_dict(shoulders_and_face_pdf_narrow_memory, elem, shoulders_and_face_narrow_pdf),
            map(tuple, small_image_row_stacked)), dtype=float)
    small_mask_only_narrow_foreground_row_stacked = (
                shoulders_face_narrow_probabilities > 0.1 * np.mean(shoulders_face_narrow_probabilities)).astype(
        np.uint8)

    small_mask_only_foreground_2d = small_mask_only_narrow_foreground_row_stacked.reshape(
        (SHOULDERS_HEIGHT - OVERHEAD_HEIGHT, right_index - left_index + 1))
    omega_b_around_face_indices = choose_indices_for_background(small_mask_only_foreground_2d, 300)
    ''' COMPARE FOREGROUND WITH BACKGROUND '''
    shoulders_face_wide_probabilities = np.fromiter(
        map(lambda elem: check_in_dict(shoulders_and_face_pdf_wide_memory, elem, shoulders_and_face_wide_pdf),
            map(tuple, small_image_row_stacked)), dtype=float)

    background_around_face_pdf = estimate_pdf(
        original_frame=original_frame[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1],
        indices=omega_b_around_face_indices, bw_method=1)
    background_around_face_probabilites = background_around_face_pdf(small_image_row_stacked)

    small_mask_probs_stacked = (shoulders_face_wide_probabilities > background_around_face_probabilites).astype(
        np.uint8)

    final_small_mask_stacked = small_mask_probs_stacked * small_mask_only_narrow_foreground_row_stacked

    final_small_mask_2d = final_small_mask_stacked.reshape(
        (SHOULDERS_HEIGHT - OVERHEAD_HEIGHT, right_index - left_index + 1))
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = final_small_mask_2d
    cv2.imwrite(f'face_frame_{frame_index}.png', scale_matrix_0_to_255(person_mask))
    cv2.imwrite(f'face_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))


    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.morphologyEx(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8), iterations=2)
    cv2.imwrite(f'face_FIRST_CLOSE_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))


    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.ones((5, 1)), iterations=1)
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.ones((1, 5)), iterations=1)
    cv2.imwrite(f'face_erode_frame_{frame_index}.png', scale_matrix_0_to_255(person_mask))
    cv2.imwrite(f'face_erode_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))

    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.identity(3,dtype=np.uint8), iterations=1)
    inverse_identity = np.zeros((3,3),dtype=np.uint8)
    inverse_identity[2,0]=1
    inverse_identity[1,1]=1
    inverse_identity[0,2]=1

    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.erode(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], inverse_identity, iterations=1)
    cv2.imwrite(f'face_diagonal_erode_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))

    img, contours, hierarchy = cv2.findContours(person_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img = np.copy(original_frame)
    cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
    contour_mask = np.zeros(person_mask.shape)
    cv2.fillPoly(contour_mask, pts=[contours[0]], color=1)
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = contour_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1]
    cv2.imwrite(f'contours_first_shot_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))

    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.dilate(
        person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], np.ones((3,3),dtype=np.uint8),
        iterations=1)

    cv2.imwrite(f'contours_first_shot_after_dilate{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))

    return person_mask