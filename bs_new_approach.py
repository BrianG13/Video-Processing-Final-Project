from constants import NUM_SAMPLES_FOR_KDE, BW_WIDE, BW_MEDIUM, SHOES_HEIGHT, SHOULDERS_HEIGHT, LEGS_HEIGHT
from utils import get_video_files, load_entire_video, scale_matrix_0_to_255, apply_mask_on_color_frame, write_video, \
    release_video_files, disk_kernel, choose_indices_for_foreground, choose_indices_for_background, estimate_pdf, \
    new_estimate_pdf, check_in_dict
import cv2
import numpy as np
from scipy import ndimage, misc

BLUE_MASK_THR = 140
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 1000
FACE_WINDOW_HEIGHT = 250
FACE_WINDOW_WIDTH = 300

def background_substraction(input_video_path):
    # Read input video
    cap, w, h, fps = get_video_files(path=input_video_path)
    # Get frame count
    frames_bgr = load_entire_video(cap, color_space='bgr')
    frames_hsv = load_entire_video(cap, color_space='hsv')
    n_frames = len(frames_bgr)

    '''COMMENTED OUT WHEN REMOVING LOADING HACK! LOOK FOR TODO'''
    # backSub = cv2.createBackgroundSubtractorKNN()
    # mask_list = np.zeros((n_frames,h,w))
    # for j in range(8):
    #     print(j)
    #     for index_frame, frame in enumerate(frames_hsv):
    #         # _, frame_s, _ = cv2.split(frame)
    #         frame_sv = frame[:,:,1:]
    #         fgMask = backSub.apply(frame_sv)
    #         fgMask = (fgMask > 200).astype(np.uint8)
    #         mask_list[index_frame] = fgMask


    # mask_list = np.load('knn_masks.np',allow_pickle=True).astype(np.uint8)  # TODO - REMOVE LOADING HACK!
    # contours_colors_list = []

    '''COMMENTED OUT WHEN REMOVING LOADING HACK! LOOK FOR TODO with (*)'''
    # omega_f_colors, omega_b_colors = None, None
    # omega_f_shoes_colors, omega_b_shoes_colors = None,None
    # person_and_blue_mask_list = np.zeros((n_frames,h,w))
    # for frame_index,frame in enumerate(frames_bgr):
    #     print(frame_index)
    #     blue_frame,_,_ = cv2.split(frame)
    #     mask_for_frame = mask_list[frame_index]
    #     mask_for_frame = cv2.morphologyEx(mask_for_frame, cv2.MORPH_CLOSE, disk_kernel(6))
    #     mask_for_frame = cv2.medianBlur(mask_for_frame, 7)
    #     _, contours, _ = cv2.findContours(mask_for_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contours.sort(key=cv2.contourArea, reverse=True)
    #     person_mask = np.zeros(mask_for_frame.shape)
    #     cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
    #     blue_mask = (blue_frame < BLUE_MASK_THR).astype(np.uint8)
    #     person_and_blue_mask = (person_mask*blue_mask).astype(np.uint8)
    #     omega_f_indices = choose_indices_for_foreground(person_and_blue_mask, 20)
    #     omega_b_indices = choose_indices_for_background(person_and_blue_mask, 20)
    #     '''Collect colors for shoes'''
    #     shoes_mask = np.copy(person_and_blue_mask)
    #     shoes_mask[:SHOES_HEIGHT,:] = 0
    #     omega_f_shoes_indices = choose_indices_for_foreground(shoes_mask, 20)
    #     shoes_mask = np.copy(person_and_blue_mask)
    #     shoes_mask[:SHOES_HEIGHT-120,:] = 1
    #     omega_b_shoes_indices = choose_indices_for_background(shoes_mask, 20)
    #     cv2.imwrite(f'after_person_and_blue_mask_and_contour_{frame_index}.png',apply_mask_on_color_frame(frame,person_and_blue_mask))
    #     person_and_blue_mask_list[frame_index] = person_and_blue_mask
    #     if omega_f_colors is None:
    #         omega_f_colors = frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]
    #         omega_b_colors = frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]
    #         omega_f_shoes_colors = frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]
    #         omega_b_shoes_colors = frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]
    #
    #     else:
    #         omega_f_colors = np.concatenate((omega_f_colors,frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]))
    #         omega_b_colors = np.concatenate((omega_b_colors,frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]))
    #         omega_f_shoes_colors = np.concatenate((omega_f_shoes_colors,frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]))
    #         omega_b_shoes_colors = np.concatenate((omega_b_shoes_colors,frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]))
    #
    # person_and_blue_mask_list.dump('person_and_blue_mask_list.npy')
    # omega_f_colors.dump('omega_f_colors.npy')
    # omega_b_colors.dump('omega_b_colors.npy')
    # omega_f_shoes_colors.dump('omega_f_shoes_colors.npy')
    # omega_b_shoes_colors.dump('omega_b_shoes_colors.npy')
    # exit()

    # person_and_blue_mask_list = np.load('person_and_blue_mask_list.npy',allow_pickle=True)  # TODO(*) - LOADING HACK! REMOVE
    # omega_f_colors = np.load('omega_f_colors.npy',allow_pickle=True)  # TODO(*) - LOADING HACK! REMOVE
    # omega_b_colors = np.load('omega_b_colors.npy',allow_pickle=True)  # TODO(*) - LOADING HACK! REMOVE
    # omega_f_shoes_colors = np.load('omega_f_shoes_colors.npy',allow_pickle=True)  # TODO(*) - LOADING HACK! REMOVE
    # omega_b_shoes_colors = np.load('omega_b_shoes_colors.npy',allow_pickle=True)  # TODO(*) - LOADING HACK! REMOVE
    #
    # foreground_pdf = new_estimate_pdf(omega_values=omega_f_colors, bw_method=BW_MEDIUM)
    # background_pdf = new_estimate_pdf(omega_values=omega_b_colors, bw_method=BW_MEDIUM)
    # foreground_shoes_pdf = new_estimate_pdf(omega_values=omega_f_shoes_colors, bw_method=BW_MEDIUM)
    # background_shoes_pdf = new_estimate_pdf(omega_values=omega_b_shoes_colors, bw_method=BW_MEDIUM)
    #
    # foreground_pdf_memoization, background_pdf_memoization = dict(), dict()
    # foreground_shoes_pdf_memoization, background_shoes_pdf_memoization = dict(), dict()
    # or_mask_list = np.zeros((n_frames,h,w))
    # for frame_index,frame in enumerate(frames_bgr):
    #     print(frame_index)
    #     person_and_blue_mask = person_and_blue_mask_list[frame_index]
    #     person_and_blue_mask_indices = np.where(person_and_blue_mask == 1)
    #     y_mean,x_mean = int(np.mean(person_and_blue_mask_indices[0])), int(np.mean(person_and_blue_mask_indices[1]))
    #     small_frame_bgr = frame[max(0,y_mean-WINDOW_HEIGHT//2):min(h,y_mean+WINDOW_HEIGHT//2),
    #                             max(0,x_mean-WINDOW_WIDTH//2):min(w,x_mean+WINDOW_WIDTH//2),
    #                             :]
    #     small_person_and_blue_mask = person_and_blue_mask[max(0,y_mean-WINDOW_HEIGHT//2):min(h,y_mean+WINDOW_HEIGHT//2),
    #                                     max(0,x_mean-WINDOW_WIDTH//2):min(w,x_mean+WINDOW_WIDTH//2)]
    #
    #     small_person_and_blue_mask_indices = np.where(small_person_and_blue_mask == 1)
    #     small_probs_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
    #     small_foreground_probabilities_stacked = np.fromiter(
    #         map(lambda elem: check_in_dict(foreground_pdf_memoization, elem, foreground_pdf),
    #             map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
    #     small_background_probabilities_stacked = np.fromiter(
    #         map(lambda elem: check_in_dict(background_pdf_memoization, elem, background_pdf),
    #             map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
    #     small_probs_fg_bigger_bg_mask[small_person_and_blue_mask_indices] = (small_foreground_probabilities_stacked > small_background_probabilities_stacked).astype(np.uint8)
    #
    #     cv2.imwrite(f'after_kde_before_shoes_{frame_index}.png',apply_mask_on_color_frame(small_frame_bgr, small_probs_fg_bigger_bg_mask))
    #
    #     '''Shoes restoration'''
    #     smaller_upper_white_mask = np.copy(small_probs_fg_bigger_bg_mask)
    #     smaller_upper_white_mask[:-270,:] = 1
    #     small_probs_fg_bigger_bg_mask_black_indices = np.where(smaller_upper_white_mask == 0)
    #     small_probs_shoes_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
    #     small_shoes_foreground_probabilities_stacked = np.fromiter(
    #         map(lambda elem: check_in_dict(foreground_shoes_pdf_memoization, elem, foreground_shoes_pdf),
    #             map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)
    #
    #     small_shoes_background_probabilities_stacked = np.fromiter(
    #         map(lambda elem: check_in_dict(background_shoes_pdf_memoization, elem, background_shoes_pdf),
    #             map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)
    #
    #     shoes_fg_shoes_bg_ratio = small_shoes_foreground_probabilities_stacked / (small_shoes_foreground_probabilities_stacked + small_shoes_background_probabilities_stacked)
    #     shoes_fg_beats_shoes_bg_mask = (shoes_fg_shoes_bg_ratio > 0.75).astype(np.uint8)
    #     small_probs_shoes_fg_bigger_bg_mask[small_probs_fg_bigger_bg_mask_black_indices] = shoes_fg_beats_shoes_bg_mask
    #     small_probs_shoes_fg_bigger_bg_mask_indices = np.where(small_probs_shoes_fg_bigger_bg_mask==1)
    #     y_shoes_mean,x_shoes_mean = int(np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[0])),int(np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[1]))
    #     # image = np.copy(small_frame_bgr)
    #     # image = cv2.circle(image, (x_shoes_mean, y_shoes_mean), 5, (0, 255, 0), 2)
    #     # # Displaying the image
    #     # cv2.imshow('sas', image)
    #     # cv2.waitKey(0)
    #     small_or_mask = np.zeros(small_probs_fg_bigger_bg_mask.shape)
    #     DELTA_Y = 0
    #     small_or_mask[:y_shoes_mean-DELTA_Y,:] = small_probs_fg_bigger_bg_mask[:y_shoes_mean-DELTA_Y,:]
    #     small_or_mask[y_shoes_mean-DELTA_Y:,:] = np.maximum(small_probs_fg_bigger_bg_mask[y_shoes_mean-DELTA_Y:,:],small_probs_shoes_fg_bigger_bg_mask[y_shoes_mean-DELTA_Y:,:]).astype(np.uint8)
    #
    #     cv2.imwrite(f'after_kde_after_shoes_{frame_index}.png',apply_mask_on_color_frame(small_frame_bgr, small_or_mask))
    #
    #     small_or_mask[y_shoes_mean-DELTA_Y-30:,:] = cv2.morphologyEx(small_or_mask[y_shoes_mean-DELTA_Y-30:,:], cv2.MORPH_CLOSE, np.ones((1,20)))
    #     small_or_mask[y_shoes_mean-DELTA_Y-30:,:] = cv2.morphologyEx(small_or_mask[y_shoes_mean-DELTA_Y-30:,:], cv2.MORPH_CLOSE, disk_kernel(20))
    #     cv2.imwrite(f'after_morphologic_{frame_index}.png',apply_mask_on_color_frame(small_frame_bgr, small_or_mask))
    #
    #     or_mask = np.zeros(person_and_blue_mask.shape)
    #     or_mask[max(0,y_mean-WINDOW_HEIGHT//2):min(h,y_mean+WINDOW_HEIGHT//2),
    #                                     max(0,x_mean-WINDOW_WIDTH//2):min(w,x_mean+WINDOW_WIDTH//2)] = small_or_mask
    #     or_mask_list[frame_index] = or_mask
    #
    # or_mask_list.dump('or_mask_list.npy')  # TODO - REMOVE! DUMP HACK!
    # exit()

    or_mask_list = np.load('or_mask_list.npy',allow_pickle=True)  # TODO - REMOVE! LOAD HACK!
    omega_f_face_colors, omega_b_face_colors = None, None
    # for frame_index,frame in enumerate(frames_bgr):
    #     print(frame_index)
    #     or_mask = or_mask_list[frame_index]
    #     face_mask = np.copy(or_mask)
    #     face_mask[SHOULDERS_HEIGHT:,:] = 0
    #     face_mask_indices = np.where(face_mask == 1)
    #     y_mean,x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))
    #
    #     small_frame_bgr = frame[max(0,y_mean-FACE_WINDOW_HEIGHT//2):min(h,y_mean+FACE_WINDOW_HEIGHT//2),
    #                             max(0,x_mean-FACE_WINDOW_WIDTH//2):min(w,x_mean+FACE_WINDOW_WIDTH//2),
    #                             :]
    #     face_mask = np.copy(or_mask)  # Restore face mask again
    #     small_face_mask = face_mask[max(0,y_mean-FACE_WINDOW_HEIGHT//2):min(h,y_mean+FACE_WINDOW_HEIGHT//2),
    #                                     max(0,x_mean-FACE_WINDOW_WIDTH//2):min(w,x_mean+FACE_WINDOW_WIDTH//2)]
    #
    #     small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((20,1),np.uint8))
    #     small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((1,20),np.uint8))
    #     cv2.imshow('ssa',apply_mask_on_color_frame(small_frame_bgr,small_face_mask))
    #     cv2.waitKey(0)
    #     continue
    #
    #     omega_f_face_indices = choose_indices_for_foreground(small_face_mask, 20)
    #     omega_b_face_indices = choose_indices_for_background(small_face_mask, 20)
    #     if omega_f_face_colors is None:
    #         omega_f_face_colors = small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]
    #         omega_b_face_colors = small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]
    #     else:
    #         omega_f_face_colors = np.concatenate((omega_f_face_colors,small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]))
    #         omega_b_face_colors = np.concatenate((omega_b_face_colors,small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]))

    # omega_f_face_colors.dump('omega_f_face_colors.npy')
    # omega_b_face_colors.dump('omega_b_face_colors.npy')
    # exit()
    omega_f_face_colors = np.load('omega_f_face_colors.npy',allow_pickle=True)
    omega_b_face_colors = np.load('omega_b_face_colors.npy',allow_pickle=True)

    foreground_face_pdf = new_estimate_pdf(omega_values=omega_f_face_colors, bw_method=BW_MEDIUM)
    background_face_pdf = new_estimate_pdf(omega_values=omega_b_face_colors, bw_method=BW_MEDIUM)
    foreground_face_pdf_memoization, background_face_pdf_memoization = dict(), dict()
    for frame_index,frame in enumerate(frames_bgr):
        print(frame_index)
        or_mask = or_mask_list[frame_index]
        face_mask = np.copy(or_mask)
        face_mask[SHOULDERS_HEIGHT:,:] = 0
        face_mask_indices = np.where(face_mask == 1)
        y_mean,x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))

        small_frame_bgr = frame[max(0,y_mean-FACE_WINDOW_HEIGHT//2):min(h,y_mean+FACE_WINDOW_HEIGHT//2),
                                max(0,x_mean-FACE_WINDOW_WIDTH//2):min(w,x_mean+FACE_WINDOW_WIDTH//2),
                                :]
        face_mask = np.copy(or_mask)  # Restore face mask again
        small_face_mask = face_mask[max(0,y_mean-FACE_WINDOW_HEIGHT//2):min(h,y_mean+FACE_WINDOW_HEIGHT//2),
                                        max(0,x_mean-FACE_WINDOW_WIDTH//2):min(w,x_mean+FACE_WINDOW_WIDTH//2)]

        small_frame_bgr_stacked = small_frame_bgr.reshape((-1,3))

        small_face_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(foreground_face_pdf_memoization, elem, foreground_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)
        small_face_background_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(background_face_pdf_memoization, elem, background_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)

        # small_probs_face_fg_bigger_face_bg_mask_stacked = \
        #     (small_face_foreground_probabilities_stacked > small_face_background_probabilities_stacked).astype(np.uint8)

        small_face_foreground_probabilities = small_face_foreground_probabilities_stacked.reshape(small_face_mask.shape)
        small_face_background_probabilities = small_face_background_probabilities_stacked.reshape(small_face_mask.shape)
        '''conv try'''

        small_face_foreground_probabilities = ndimage.median_filter(small_face_foreground_probabilities, size=20)
        small_face_background_probabilities = ndimage.median_filter(small_face_background_probabilities, size=20)
        '''end'''
        small_probs_face_fg_bigger_face_bg_mask = (small_face_foreground_probabilities > small_face_background_probabilities).astype(np.uint8)
        small_probs_face_fg_bigger_face_bg_mask = cv2.morphologyEx(small_probs_face_fg_bigger_face_bg_mask, cv2.MORPH_OPEN, np.ones((5,1),np.uint8))
        cv2.imwrite(f'after_face_restore_before_contour_{frame_index}.png',apply_mask_on_color_frame(small_frame_bgr, small_probs_face_fg_bigger_face_bg_mask))
        _, contours, _ = cv2.findContours(small_probs_face_fg_bigger_face_bg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        small_contour_mask = np.zeros(small_probs_face_fg_bigger_face_bg_mask.shape,dtype=np.uint8)
        cv2.fillPoly(small_contour_mask, pts=[contours[0]], color=1)
        cv2.imwrite(f'after_face_restore_after_contour_{frame_index}.png',apply_mask_on_color_frame(small_frame_bgr, small_contour_mask))



    # write_video(output_path='knn_contours_after_close6_and_median7_frames_sv_and_blue.avi', frames=contours_colors_list, fps=fps, out_size=(w,h), is_color=True)
    # release_video_files(cap)

background_substraction('stabilize.avi')
# knn_result = cv2.imread('knn_frame_140_7.png',cv2.IMREAD_GRAYSCALE)

# cap, w, h, fps = get_video_files(path='stabilize.avi')
# frames_bgr = load_entire_video(cap, color_space='bgr')
# input_mask = cv2.imread('mask_for_frame_70.png',cv2.IMREAD_GRAYSCALE)
# input_mask = (input_mask > 200).astype(np.uint8)
# for i in range(3,15,2):
#     x = disk_kernel(i-1)
#     new_mask = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, disk_kernel(i-1))
#     new_mask = cv2.medianBlur(new_mask, i)
#     cv2.imshow('ss',apply_mask_on_color_frame(frames_bgr[70], new_mask))
#     cv2.waitKey(0)
