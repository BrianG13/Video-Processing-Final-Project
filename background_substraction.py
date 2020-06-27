# import cv2
# import numpy as np
#
# from constants import (
#     FRAME_INDEX_FOR_PERSON_KDE,
#     NUM_SAMPLES_FOR_KDE,
#     BW_WIDE,
#     BW_MEDIUM,
#     BW_NARROW,
#     LEGS_HEIGHT,
#     FRAME_INDEX_FOR_FACE_TUNING
# )
#
# from fine_tune_background_substraction import (
#     fine_tune_contour_mask,
#     restore_shoes,
#     build_shoulders_face_pdf,
#     build_shoes_pdf,
#     remove_signs
# )
# from utils import (
#     get_video_files,
#     release_video_files,
#     write_video,
#     load_entire_video,
#     scale_matrix_0_to_255,
#     apply_mask_on_color_frame,
#     choose_indices_for_foreground,
#     choose_indices_for_background,
#     check_in_dict,
#     estimate_pdf,
# )
#
#
# def background_substraction(input_video_path):
#     # Read input video
#     cap, w, h, fps = get_video_files(path=input_video_path)
#     # Get frame count
#     frames_bgr = load_entire_video(cap, color_space='bgr')
#     frames_hsv = load_entire_video(cap, color_space='hsv')
#     n_frames = len(frames_bgr)
#
#     "Find medians in colors space "
#     medians_frame_bgr = np.median(frames_bgr, axis=0)
#     medians_frame_b, medians_frame_g, _ = cv2.split(medians_frame_bgr)
#     medians_frame_hsv = np.median(frames_hsv, axis=0)
#     _, medians_frame_s, medians_frame_v = cv2.split(medians_frame_hsv)
#
#     mask_fine_tuned_after_contours_list, removed_signs_color_frame_list, removed_signs_mask_list = [], [], []
#     '''Build KDE from specified frame filtered by s,v channels'''
#     mask_for_building_kde = build_person_mask_for_kde(frames_bgr[FRAME_INDEX_FOR_PERSON_KDE], medians_frame_s,
#                                                       medians_frame_v)
#     omega_f_indices = choose_indices_for_foreground(mask_for_building_kde, NUM_SAMPLES_FOR_KDE)
#     omega_b_indices = choose_indices_for_background(mask_for_building_kde, NUM_SAMPLES_FOR_KDE)
#     foreground_pdf = estimate_pdf(original_frame=frames_bgr[FRAME_INDEX_FOR_PERSON_KDE],
#                                   indices=omega_f_indices, bw_method=BW_WIDE)
#     background_pdf = estimate_pdf(original_frame=frames_bgr[FRAME_INDEX_FOR_PERSON_KDE],
#                                   indices=omega_b_indices, bw_method=BW_WIDE)
#
#     foreground_pdf_memoization, background_pdf_memoization = dict(), dict()
#     ''''''
#     for frame_index, frame in enumerate(frames_bgr):
#         print(f"[BS] - Frame: {frame_index} / {n_frames}")
#
#         ''' Get probability for each pixel to be in bg or fg '''
#         row_stacked_original_frame = frame.reshape((h * w), 3)
#         foreground_probabilities = np.fromiter(
#             map(lambda elem: check_in_dict(foreground_pdf_memoization, elem, foreground_pdf),
#                 map(tuple, row_stacked_original_frame)), dtype=float)
#         foreground_probabilities = foreground_probabilities.reshape((h, w))
#         background_probabilities = np.fromiter(
#             map(lambda elem: check_in_dict(background_pdf_memoization, elem, background_pdf),
#                 map(tuple, row_stacked_original_frame)), dtype=float)
#         background_probabilities = background_probabilities.reshape((h, w))
#
#         probs_fg_bigger_bg_mask = (foreground_probabilities > background_probabilities).astype(
#             np.uint8)  # TODO - MAYBE CAUSE PROBLEMS
#         probs_fg_bigger_bg_mask = cv2.erode(probs_fg_bigger_bg_mask, np.ones((3, 1), np.uint8), iterations=2)
#         probs_fg_bigger_bg_mask = cv2.erode(probs_fg_bigger_bg_mask, np.ones((1, 3), np.uint8), iterations=1)
#         probs_fg_bigger_bg_mask = cv2.morphologyEx(probs_fg_bigger_bg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
#         probs_fg_bigger_bg_mask = cv2.morphologyEx(probs_fg_bigger_bg_mask, cv2.MORPH_CLOSE,
#                                                    np.identity(10, np.uint8))  # Connect detached hands
#
#         _, contours, _ = cv2.findContours(probs_fg_bigger_bg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contours.sort(key=cv2.contourArea, reverse=True)
#
#         contour_mask = np.zeros((h, w))
#         cv2.fillPoly(contour_mask, pts=[contours[0]], color=1)
#
#         mask_fine_tuned_after_contours = fine_tune_contour_mask(contour_mask=contour_mask,
#                                                                 original_frame=frame,
#                                                                 background_pdf=background_pdf,
#                                                                 background_pdf_memoization=background_pdf_memoization,
#                                                                 foreground_pdf=foreground_pdf,
#                                                                 foreground_pdf_memoization=foreground_pdf_memoization)
#
#         mask_fine_tuned_after_contours_list.append(mask_fine_tuned_after_contours)
#         cv2.imwrite(f'BS_BEFORE_SHOES_AND_SIGNS{frame_index}.png',apply_mask_on_color_frame(frames_bgr[frame_index],
#                                                                                 mask_fine_tuned_after_contours))  # TODO - DELETE
#
#
#     shoes_foreground_specialist_pdf = build_shoes_pdf(original_frame=frames_bgr[6],
#                                                       mask=mask_fine_tuned_after_contours_list[6])
#
#     shoulders_face_narrow_pdf = build_shoulders_face_pdf(
#         mask_fine_tuned_after_contours_list[FRAME_INDEX_FOR_FACE_TUNING],
#         frames_bgr[FRAME_INDEX_FOR_FACE_TUNING],
#         bw_method=BW_NARROW)
#
#     shoulders_face_wide_pdf = build_shoulders_face_pdf(mask_fine_tuned_after_contours_list[FRAME_INDEX_FOR_FACE_TUNING],
#                                                        frames_bgr[FRAME_INDEX_FOR_FACE_TUNING],
#                                                        bw_method=BW_MEDIUM)
#
#     shoulders_and_face_pdf_narrow_memoization, shoulders_and_face_pdf_wide_memoization = dict(), dict()
#
#     for frame_index, frame in enumerate(frames_bgr):
#         print(f"[BS] Restoring shoes & Removing signs - Frame: {frame_index} / {n_frames}")
#         mask_after_shoes_fix = restore_shoes(contour_mask=mask_fine_tuned_after_contours_list[frame_index],
#                                              original_frame=frame,
#                                              shoes_specialist_pdf=shoes_foreground_specialist_pdf)
#
#         '''Merge shoes fix results with previous'''
#         mask_fine_tuned_with_shoes = np.copy(mask_fine_tuned_after_contours_list[frame_index])
#         mask_fine_tuned_with_shoes[LEGS_HEIGHT:, :] = mask_after_shoes_fix[LEGS_HEIGHT:, :]
#         mask_fine_tuned_with_shoes = cv2.morphologyEx(mask_fine_tuned_with_shoes, cv2.MORPH_CLOSE,
#                                                       np.ones((2, 1), np.uint8))
#
#         mask_fine_tuned_with_shoes_remove_signs = remove_signs(original_frame=frames_bgr[frame_index],
#                                                                mask=mask_fine_tuned_with_shoes,
#                                                                shoulders_and_face_narrow_pdf=shoulders_face_narrow_pdf,
#                                                                shoulders_and_face_wide_pdf=shoulders_face_wide_pdf,
#                                                                shoulders_and_face_pdf_narrow_memoization=shoulders_and_face_pdf_narrow_memoization,
#                                                                shoulders_and_face_pdf_wide_memoization=shoulders_and_face_pdf_wide_memoization
#                                                                )
#
#         removed_signs_color_frame = apply_mask_on_color_frame(frames_bgr[frame_index],
#                                                               mask_fine_tuned_with_shoes_remove_signs)
#         removed_signs_color_frame_list.append(removed_signs_color_frame)
#         removed_signs_mask_list.append(scale_matrix_0_to_255(mask_fine_tuned_with_shoes_remove_signs))
#
#         cv2.imwrite(f'BS_AFTER_SHOES_AND_SIGNS_{frame_index}.png',removed_signs_color_frame)  # TODO - DELETE
#
#     write_video('binary.avi', frames=removed_signs_mask_list, fps=fps, out_size=(w, h),
#                 is_color=False)
#     write_video('extracted.avi', frames=removed_signs_color_frame_list, fps=fps, out_size=(w, h),
#                 is_color=True)
#     release_video_files(cap)
#
#
# def build_person_mask_for_kde(frame, medians_frame_s, medians_frame_v):
#     print('[BS] - Processing data for median filtering')
#     curr_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     _, curr_s, curr_v = cv2.split(curr_hsv)
#     curr_b, _, _ = cv2.split(frame)
#     diff_s = np.abs(medians_frame_s - curr_s).astype(np.uint8)
#     diff_v = np.abs(medians_frame_v - curr_v).astype(np.uint8)
#     mask_s = (diff_s > np.mean(diff_s) * 5)
#     mask_v = (diff_v > np.mean(diff_v) * 5)
#     mask_or = (mask_s | mask_v).astype(np.uint8)
#     kernel = np.ones((7, 7), np.uint8)
#     dilation = cv2.dilate(mask_or, kernel, iterations=1)  # TODO - MAYBE CHANGE TO CLOSE!
#     # Filtering blue colors
#     blue_mask = (curr_b < 140).astype(np.uint8)
#     blue_mask = cv2.erode(blue_mask, np.ones((3, 3), np.uint8), iterations=2).astype(np.uint8)
#
#     final_mask = dilation * blue_mask
#     final_mask = cv2.erode(final_mask, np.ones((5, 5), np.uint8), iterations=4)
#     return final_mask
