import cv2
import numpy as np

from utils import (
    apply_mask_on_color_frame,
    check_in_dict,
    scale_matrix_0_to_255,
    SHOES_HEIGHT,
SHOULDERS_HEIGHT,
OVERHEAD_HEIGHT,
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

def build_shoulders_face_pdf(mask,image,bw_method):
    small_mask = mask[:SHOULDERS_HEIGHT,:]
    small_image = image[:SHOULDERS_HEIGHT,:]
    omega_f_indices = choose_indices_for_foreground(small_mask, 200)
    foreground_pdf = estimate_pdf(original_frame=small_image, indices=omega_f_indices, bw_method=bw_method)
    return foreground_pdf




def remove_signs(frame_index, original_frame, mask,shoulders_and_face_pdf,shoulders_and_face_pdf_memory):
    
    person_mask = np.copy(mask)
    shoes_indices_rectangle_x_axis = np.where(person_mask == 1)[1]
    left_index, right_index = np.min(shoes_indices_rectangle_x_axis), np.max(shoes_indices_rectangle_x_axis)
    small_mask = person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT,left_index:right_index+1]
    small_image = original_frame[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT,left_index:right_index+1]
    # small_mask_row_stacked = small_mask.reshape((-1,1))
    # small_mask_indices = np.where(small_mask_row_stacked == 1)
    small_image_row_stacked = small_image.reshape(-1, 3)

    shoulders_face_probabilities = np.fromiter(map(lambda elem: check_in_dict(shoulders_and_face_pdf_memory, elem, shoulders_and_face_pdf),
                                               map(tuple, small_image_row_stacked)), dtype=float)

    shoulders_face_mask = (shoulders_face_probabilities > 0.1*np.mean(shoulders_face_probabilities)).astype(np.uint8)

    # small_mask_row_stacked[small_mask_indices] = shoulders_face_mask
    small_mask_row_stacked = shoulders_face_mask

    small_mask_2d = small_mask_row_stacked.reshape((SHOULDERS_HEIGHT-OVERHEAD_HEIGHT,right_index-left_index+1))
    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = small_mask_2d
    cv2.imwrite(f'face_frame_{frame_index}.png', scale_matrix_0_to_255(person_mask))
    cv2.imwrite(f'face_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))

    person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.morphologyEx(person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], cv2.MORPH_OPEN,
                                                        np.ones((2, 2), np.uint8), iterations=1)
    # person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.morphologyEx(person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], cv2.MORPH_CLOSE,
    #                                                     np.ones((4, 4), np.uint8), iterations=3)
    # person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.morphologyEx(
    #     person_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1], cv2.MORPH_CLOSE,
    #     np.ones((8, 1), np.uint8), iterations=2)


    cv2.imwrite(f'face_closing_frame_{frame_index}.png', scale_matrix_0_to_255(person_mask))
    cv2.imwrite(f'face_closing_frame_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, person_mask))

    person_mask = person_mask.astype(np.uint8)
    img, contours, hierarchy = cv2.findContours(person_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img = np.copy(original_frame)
    cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
    cv2.imwrite(f'contours_face_{frame_index}.png', img)
    contour_mask = np.zeros(person_mask.shape)
    cv2.fillPoly(contour_mask, pts=[contours[0]], color=1)
    contour_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1] = cv2.dilate(contour_mask[OVERHEAD_HEIGHT:SHOULDERS_HEIGHT, left_index:right_index + 1],np.ones((3,3)),iterations=2)
    contour_mask = cv2.dilate(contour_mask,np.ones((3,3)),iterations=1)


    cv2.imwrite(f'face_contour_closed_{frame_index}_color.png',
                apply_mask_on_color_frame(original_frame, contour_mask))
