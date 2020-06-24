import cv2
import numpy as np
import GeodisTK
import time
from kernel_estimation import estimate_pdf
from utils import load_entire_video, get_video_files, choose_indices_for_foreground, choose_indices_for_background, \
    apply_mask_on_color_frame, scale_matrix_0_to_255,write_video

EPSILON = 0.99
ERODE_ITERATIONS = 6
DILATE_ITERATIONS = 3
GEODISTK_ITERATIONS = 2
REFINEMENT_WINDOW_SIZE = 20
KDE_BW = 1
R = 0.001

def video_matting(input_stabilize_video, binary_video_path, output_video_path,
                  new_background):
    START_TIME = time.time()
    # Read input video
    cap_stabilize, _, w, h, fps_stabilize = get_video_files(input_stabilize_video, output_video_path, isColor=True)
    cap_binary, _, _, _, fps_binary = get_video_files(binary_video_path, 'delete.avi', isColor=False)

    # Get frame count
    n_frames = int(cap_stabilize.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')
    frames_yuv = load_entire_video(cap_stabilize, color_space='yuv')
    frames_binary = load_entire_video(cap_binary, color_space='bw')

    '''Resize new background'''
    new_background = cv2.resize(new_background, (w, h))
    full_matted_frames_list = list()
    for frame_index in range(n_frames):
        print(frame_index)
        luma_frame, _, _ = cv2.split(frames_yuv[frame_index])
        bgr_frame = frames_bgr[frame_index]

        original_mask_frame = frames_binary[frame_index]
        original_mask_frame = (original_mask_frame > 150).astype(np.uint8)

        '''Find indices for resizing image to work only on relevant part!'''
        binary_frame_rectangle_x_axis = np.where(original_mask_frame == 1)[1]
        left_index, right_index = np.min(binary_frame_rectangle_x_axis), np.max(binary_frame_rectangle_x_axis)
        left_index, right_index = max(0, left_index - 50), min(right_index + 50, original_mask_frame.shape[1] - 1)
        binary_frame_rectangle_y_axis = np.where(original_mask_frame == 1)[0]
        top_index, bottom_index = np.min(binary_frame_rectangle_y_axis), np.max(binary_frame_rectangle_y_axis)
        top_index, bottom_index = max(0, top_index - 50), min(bottom_index + 50, original_mask_frame.shape[0] - 1)
        ''' Resize images '''
        smaller_luma_frame = luma_frame[top_index:bottom_index, left_index:right_index]
        smaller_bgr_frame = bgr_frame[top_index:bottom_index, left_index:right_index]
        smaller_new_background = new_background[top_index:bottom_index, left_index:right_index]
        '''Eroded foreground mask option'''
        foreground_mask = cv2.erode(original_mask_frame, np.ones((3, 3)), iterations=ERODE_ITERATIONS)
        # cv2.imwrite(f'foreground_scrible_{frame_index}.png',
        #             apply_mask_on_color_frame(frames_bgr[frame_index], foreground_mask))

        '''Resize foreground mask & Build distance map for foreground'''
        smaller_foreground_mask = foreground_mask[top_index:bottom_index, left_index:right_index]
        smaller_foreground_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_foreground_mask,
                                                                          1.0, GEODISTK_ITERATIONS)
        # cv2.imwrite(f'foreground_distmap_{frame_index}.png',
        #             cv2.cvtColor(smaller_foreground_distance_map, cv2.COLOR_GRAY2BGR))

        '''Resize image & Build distance map for foreground'''
        background_mask = cv2.dilate(original_mask_frame, np.ones((3, 3)), iterations=DILATE_ITERATIONS)
        # cv2.imwrite(f'background_scrible_{frame_index}.png',
        #             apply_mask_on_color_frame(frames_bgr[frame_index], background_mask))
        background_mask = 1 - background_mask
        smaller_background_mask = background_mask[top_index:bottom_index, left_index:right_index]
        smaller_background_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_background_mask,
                                                                          1.0, GEODISTK_ITERATIONS)
        # cv2.imwrite(f'background_distmap_{frame_index}.png', smaller_background_distance_map)


        ''' Building narrow band undecided zone'''
        smaller_foreground_distance_map = smaller_foreground_distance_map / (smaller_foreground_distance_map + smaller_background_distance_map)
        smaller_background_distance_map = 1-smaller_foreground_distance_map
        smaller_narrow_band_mask = (np.abs(smaller_foreground_distance_map - smaller_background_distance_map) < EPSILON).astype(np.uint8)
        smaller_narrow_band_mask_indices = np.where(smaller_narrow_band_mask == 1)
        smaller_undecided_image = np.copy(smaller_bgr_frame)
        smaller_undecided_image[smaller_narrow_band_mask_indices] = np.asarray([0, 255, 0])
        # cv2.imwrite(f'epsilon_color_{frame_index}.png', smaller_undecided_image)

        smaller_decided_foreground_mask = (smaller_foreground_distance_map < smaller_background_distance_map - EPSILON).astype(np.uint8)
        smaller_decided_background_mask = (smaller_background_distance_map >= smaller_foreground_distance_map - EPSILON).astype(np.uint8)
        omega_f_indices = choose_indices_for_foreground(smaller_decided_foreground_mask, 200)
        omega_b_indices = choose_indices_for_background(smaller_decided_background_mask, 200)
        foreground_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_f_indices, bw_method=KDE_BW)
        background_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_b_indices, bw_method=KDE_BW)
        smaller_narrow_band_foreground_probs = foreground_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])
        smaller_narrow_band_background_probs = background_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])
        w_f = np.power(smaller_foreground_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_foreground_probs
        w_b = np.power(smaller_background_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_background_probs
        alpha_narrow_band = w_f / (w_f + w_b)
        smaller_alpha = np.copy(smaller_decided_foreground_mask).astype(np.float)
        smaller_alpha[smaller_narrow_band_mask_indices] = alpha_narrow_band

        smaller_matted_frame = smaller_alpha[:,:,np.newaxis] * smaller_bgr_frame + (1 - smaller_alpha[:,:,np.newaxis]) * smaller_new_background
        # cv2.imwrite(f'smaller_matted_frame_{frame_index}.png',smaller_matted_frame)


        ''' Narrow band - refinement - matting - solving argmin problem '''
        # for i in range(len(smaller_undecided_mask_indices[0])):
        #     print(f'Fixing pixel: {i} / {len(smaller_undecided_mask_indices[0])}')
        #     pixel_coords = (smaller_undecided_mask_indices[0][i], smaller_undecided_mask_indices[1][i])
        #     foreground_neighbors_indices, background_neighbors_indices = get_foreground_and_background_neighbors(
        #                 pixel_coords=pixel_coords,
        #                 smaller_decided_background_mask=smaller_decided_background_mask,
        #                 smaller_decided_foreground_mask=smaller_decided_foreground_mask,
        #                 window_size=REFINEMENT_WINDOW_SIZE)
        #
        #     foreground_best_neighbor_indices, background_best_neighbor_indices = find_best_couple(pixel_coords,
        #                                                                                             smaller_bgr_frame,
        #                                                                                           smaller_new_background,
        #                                                                                           smaller_alpha,
        #                                                                                           foreground_neighbors_indices,
        #                                                                                           background_neighbors_indices
        #                                                                                           )
        #     alpha_x = smaller_alpha[pixel_coords]
        #     smaller_matted_frame[pixel_coords] = alpha_x*smaller_bgr_frame[foreground_best_neighbor_indices] + \
        #                                             (1-alpha_x)*smaller_new_background[pixel_coords]

        full_matted_frame = np.copy(new_background)
        full_matted_frame[top_index:bottom_index, left_index:right_index] = smaller_matted_frame
        full_matted_frames_list.append(full_matted_frame)
        # cv2.imwrite(f'after_matting_{frame_index}.png', full_matted_frame)

    write_video(output_path='matted_video.avi', frames=full_matted_frames_list, fps=fps_stabilize, out_size=(w,h), is_color=True)
    print(f'ENTIRE PROCESS TIME: {time.time()-START_TIME}')

def find_best_couple(pixel_coords,smaller_bgr_frame,smaller_new_background,smaller_alpha,foreground_neighbors_indices,background_neighbors_indices):
    original_color = smaller_bgr_frame[pixel_coords]
    alpha_x = smaller_alpha[pixel_coords]
    color_xf = smaller_bgr_frame[foreground_neighbors_indices]
    color_xb = smaller_new_background[background_neighbors_indices]
    repeated_xf = np.repeat(color_xf, repeats=color_xb.shape[0], axis=0)
    repeated_xb = np.resize(np.stack([color_xb for _ in range(color_xf.shape[0])], axis=0),(color_xf.shape[0]*color_xb.shape[0],3))
    original_color_matrix = np.ones((color_xf.shape[0]*color_xb.shape[0],3))*original_color
    norm_big_matrix = np.linalg.norm(alpha_x*repeated_xf + (1-alpha_x)*repeated_xb - original_color_matrix)
    minimum_index = np.argmin(norm_big_matrix)
    best_background_index = minimum_index % color_xb.shape[0]
    best_foreground_index = minimum_index // color_xb.shape[0]
    foreground_best_coord = (foreground_neighbors_indices[0][best_foreground_index] , foreground_neighbors_indices[1][best_foreground_index])
    background_best_coord = (background_neighbors_indices[0][best_background_index] , background_neighbors_indices[1][best_background_index])

    return foreground_best_coord, background_best_coord

def get_foreground_and_background_neighbors(pixel_coords, smaller_decided_background_mask,
                                            smaller_decided_foreground_mask,
                                            window_size):
    max_y_index, max_x_index = smaller_decided_background_mask.shape
    top_left_coords = max(pixel_coords[0] - window_size // 2, 0), max(pixel_coords[1] - window_size // 2, 0)
    bottom_right_coords = min(pixel_coords[0] + window_size // 2, max_y_index - 1), min(
        pixel_coords[1] + window_size // 2, max_x_index - 1)
    window_mask = np.ones(smaller_decided_foreground_mask.shape)
    window_mask[:top_left_coords[0], :] = 0
    window_mask[:, :top_left_coords[1]] = 0
    window_mask[:, bottom_right_coords[1]:] = 0
    window_mask[bottom_right_coords[0]:, :] = 0

    foreground_neighbours = smaller_decided_foreground_mask * window_mask
    foreground_neighbours_indices = np.where(foreground_neighbours == 1)
    background_neighbours = smaller_decided_background_mask * window_mask
    background_neighbours_indices = np.where(background_neighbours == 1)
    return foreground_neighbours_indices, background_neighbours_indices


def gamma_correction(x, gamma):
    return x ** gamma
