import cv2
import numpy as np
import GeodisTK
import matplotlib.pyplot as plt
from PIL import Image


from utils import load_entire_video, get_video_files, choose_indices_for_foreground, choose_indices_for_background, \
    apply_mask_on_color_frame, scale_matrix_0_to_255

EPSILON = 0.4
ERODE_ITERATIONS = 9
DILATE_ITERATIONS = 3
GEODISTK_ITERATIONS = 2


# beach = cv2.imread('beach.png',cv2.IMREAD_GRAYSCALE)
# x = cv2.imread('original.png',cv2.IMREAD_GRAYSCALE)
# beach = cv2.resize(beach,(x.shape[1],x.shape[0]))
# cv2.imwrite('beach.png',beach)
# cutout(
#     # input image path
#     "original.png",
#     # input trimap path
#     "trimap.png",
#     # output cutout path
#     "lemur_cutout.png",
# )
# 
# from pymatting import *
# 
# scale = 1.0
# 
# image = load_image("original.png", "RGB", scale, "box")
# trimap = load_image("trimap.png", "GRAY", scale, "nearest")
# 
# # estimate alpha from image and trimap
# alpha = estimate_alpha_cf(image, trimap)
# 
# # load new background
# new_background = load_image("beach.png", "RGB", scale, "box")
# 
# # estimate foreground from image and alpha
# foreground, background = estimate_foreground_ml(image, alpha, return_background=True)
# 
# # blend foreground with background and alpha
# new_image = blend(foreground, new_background, alpha)
# 
# # save results in a grid
# images = [image, trimap, alpha, new_image]
# grid = make_grid(images)
# save_image("lemur_at_the_beach.png", grid)
def video_matting(input_stabilize_video, binary_video_path, output_video_path):
    # Read input video
    cap_stabilize, _, _, _, fps_stabilize = get_video_files(input_stabilize_video, output_video_path, isColor=True)
    cap_binary, _, _, _, fps_binary = get_video_files(binary_video_path, 'delete.avi', isColor=False)

    # Get frame count
    n_frames = int(cap_stabilize.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    _, prev = cap_stabilize.read()

    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')
    # frames_hsv = load_entire_video(cap_stabilize, color_space='hsv')
    frames_yuv = load_entire_video(cap_stabilize, color_space='yuv')
    frames_binary = load_entire_video(cap_binary, color_space='bw')

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

        '''Eroded foreground mask option'''
        # foreground_mask = cv2.erode(original_mask_frame, np.ones((3, 3)), iterations=ERODE_ITERATIONS)
        # cv2.imwrite(f'foreground_scrible_{frame_index}.png',
        #             apply_mask_on_color_frame(frames_bgr[frame_index], foreground_mask))

        '''Sampled foreground mask option'''
        foreground_mask = cv2.erode(original_mask_frame, np.ones((3, 3)), iterations=5)
        foreground_mask_indices = choose_indices_for_foreground(foreground_mask,300)
        foreground_mask = np.zeros(foreground_mask.shape,dtype=np.uint8)
        for i in range(foreground_mask_indices.shape[0]):
            foreground_mask[foreground_mask_indices[i][0],foreground_mask_indices[i][1]] = 1
        image = np.copy(frames_bgr[frame_index])
        for index in range(foreground_mask_indices.shape[0]):
            image = cv2.circle(image, (foreground_mask_indices[index][1], foreground_mask_indices[index][0]), 2, (0, 255, 0), 1)
        ## Displaying the image
        cv2.imwrite(f'foreground_mask_sample_{frame_index}.png', image)


        '''Resize foreground mask & Build distance map for foreground'''
        smaller_foreground_mask = foreground_mask[top_index:bottom_index, left_index:right_index]
        smaller_foreground_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_foreground_mask,
                                                                          1.0, GEODISTK_ITERATIONS)
        # max_result_foreground_dismap = np.max(foreground_distance_map)
        # foreground_distance_map = foreground_distance_map / max_result_foreground_dismap
        # foreground_distance_map = gamma_correction(foreground_distance_map,gamma=0.1)#np.apply_along_axis(lambda x: gamma_correction(x=x,gamma=3),result)
        # foreground_distance_map = foreground_distance_map * max_result_foreground_dismap
        cv2.imwrite(f'foreground_distmap_{frame_index}.png',cv2.cvtColor(smaller_foreground_distance_map,cv2.COLOR_GRAY2BGR))


        '''Resize image & Build distance map for foreground'''
        background_mask = cv2.dilate(original_mask_frame, np.ones((3, 3)), iterations=DILATE_ITERATIONS)
        cv2.imwrite(f'background_scrible_{frame_index}.png',apply_mask_on_color_frame(frames_bgr[frame_index], background_mask))
        background_mask = 1 - background_mask
        smaller_background_mask = background_mask[top_index:bottom_index, left_index:right_index]
        smaller_background_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_background_mask,
                                                                          1.0, GEODISTK_ITERATIONS)
        # max_result_background_dismap = np.max(background_distance_map)
        # background_distance_map = background_distance_map / max_result_background_dismap
        # background_distance_map = gamma_correction(background_distance_map,gamma=0.1)#np.apply_along_axis(lambda x: gamma_correction(x=x,gamma=3),result)
        # background_distance_map = background_distance_map * max_result_background_dismap
        cv2.imwrite(f'background_distmap_{frame_index}.png',smaller_background_distance_map)

        '''Build binary mask from dist map'''
        smaller_alpha = smaller_foreground_distance_map / (
                    smaller_background_distance_map + smaller_foreground_distance_map)
        smaller_alpha = 1-smaller_alpha
        smaller_foreground_beats_background_alpha_mask = (smaller_alpha > 0.5).astype(np.uint8)
        cv2.imwrite(f'smaller_foreground_beats_background_alpha_mask_{frame_index}.png',
                   cv2.cvtColor(apply_mask_on_color_frame(smaller_bgr_frame,smaller_foreground_beats_background_alpha_mask),
                                             cv2.COLOR_BGR2RGB))

        # smaller_background_beats_foreground_alpha = scale_matrix_0_to_255(smaller_alpha < 0.5-EPSILON)
        # plt.imshow(smaller_background_beats_foreground_alpha)
        # plt.title(f'background_beats_foreground_alpha - frame {frame_index}')
        # plt.show()
        smaller_equal_dist_matrix_mask = (0.5 - EPSILON < smaller_alpha).astype(np.uint8) * (
                    smaller_alpha < 0.5 + EPSILON).astype(np.uint8)
        smaller_equal_dist_matrix_mask_indices = np.where(smaller_equal_dist_matrix_mask == 1)
        # smaller_bgr_frame[smaller_equal_dist_matrix_mask_indices] = np.asarray([0, 255, 0]) - TODO REMOVE
        cv2.imwrite(f'epsilon_color_{frame_index}.png',smaller_bgr_frame)





def gamma_correction(x, gamma):
    return x ** gamma
