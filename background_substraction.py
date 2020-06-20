import cv2
import numpy as np

from utils import (
    get_video_files,
    release_video_files,
    write_video,
    load_entire_video,
    scale_matrix_0_to_255,
    apply_mask_on_color_frame
)


def background_substraction(input_video_path, output_video_path):
    # Read input video
    cap, out, w, h, fps = get_video_files(input_video_path, output_video_path, isColor=True)
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    _, prev = cap.read()

    frames_bgr = load_entire_video(cap, color_space='bgr')
    frames_hsv = load_entire_video(cap, color_space='hsv')

    "Find medians in colors space "
    medians_frame_bgr = np.median(frames_bgr, axis=0)
    medians_frame_b, medians_frame_g, medians_frame_r = cv2.split(medians_frame_bgr)
    medians_frame_hsv = np.median(frames_hsv, axis=0)
    medians_frame_h, medians_frame_s, medians_frame_v = cv2.split(medians_frame_hsv)

    s_diff_from_median_list, v_diff_from_median_list, b_diff_from_median_list = [], [], []
    mask_s_list, mask_v_list, blue_mask_list, mask_or_list, mask_weighted_list = [], [], [],[], []
    original_with_or_mask_results = []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        curr_hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
        curr_h, curr_s, curr_v = cv2.split(curr_hsv)
        curr_b, curr_g, curr_r = cv2.split(curr)
        diff_s = np.abs(medians_frame_s - curr_s).astype(np.uint8)
        s_diff_from_median_list.append(diff_s)
        diff_v = np.abs(medians_frame_v - curr_v).astype(np.uint8)
        v_diff_from_median_list.append(diff_v)
        diff_b = np.abs(medians_frame_b - curr_b).astype(np.uint8)
        b_diff_from_median_list.append(diff_b)
        blue_mask = (diff_b > np.mean(diff_b) * 1.5)
        mask_s = (diff_s > np.mean(diff_s) * 5)
        mask_v = (diff_v > np.mean(diff_v) * 5)
        mask_or = (mask_s | mask_v).astype(np.uint8)

        weighted_mask = scale_matrix_0_to_255(
            (0.5 * (diff_s - np.mean(diff_s) * 5) + 0.5 * (diff_v - np.mean(diff_v) * 7) > 0).astype(np.uint8))

        kernel = np.ones((7, 7), np.uint8)
        dilation = cv2.dilate(mask_or, kernel, iterations=1)

        frame_after_or_flt = apply_mask_on_color_frame(curr,dilation)
        original_with_or_mask_results.append(frame_after_or_flt)

        mask_s = scale_matrix_0_to_255(mask_s)
        mask_s_list.append(mask_s)
        mask_v = scale_matrix_0_to_255(mask_v)
        mask_v_list.append(mask_v)
        blue_mask = scale_matrix_0_to_255(blue_mask)
        blue_mask_list.append(blue_mask)
        dilation = scale_matrix_0_to_255(dilation)
        mask_or_list.append(dilation)
        mask_weighted_list.append(weighted_mask)

    write_video('s_diff_from_median_results.avi', frames=s_diff_from_median_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('v_diff_from_median_results.avi', frames=v_diff_from_median_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('b_diff_from_median_results.avi', frames=b_diff_from_median_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('blue_mask_out.avi', frames=blue_mask_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('mask_weighted.avi', frames=blue_mask_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('mask_or.avi', frames=blue_mask_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('mask_s.avi', frames=mask_s_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('mask_v.avi', frames=mask_v_list, fps=fps, out_size=(w, h), is_color=False)
    write_video('original_with_or_mask.avi', frames=original_with_or_mask_results, fps=fps, out_size=(w, h),
                is_color=False)
    release_video_files(cap, out)


def continue_background_substraction(input_video_path, output_video_path):
    # Read input video
    cap, out = get_video_files(input_video_path, output_video_path, isColor=True)
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read first frame
    _, prev = cap.read()
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Pre-define transformation-store array
    frames_bgr = [prev]  # MEDIAN TRY
    for i in range(n_frames):
        print("Frame: " + str(i) + "/" + str(n_frames))
        # Read next frame
        success, curr = cap.read()
        if not success:
            break
        frames_bgr.append(curr)

    frames_bgr = np.asarray(frames_bgr)
    medians_frame_bgr = np.median(frames_bgr, axis=0)
    medians_frame_b, _, _ = cv2.split(medians_frame_bgr)

    out_size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    after_or_using_blue_out = cv2.VideoWriter('after_or_using_blue.avi', fourcc, fps, out_size, isColor=True)
    blue_mask_out = cv2.VideoWriter('blue_mask_out.avi', fourcc, fps, out_size, isColor=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        curr_b, _, _ = cv2.split(curr)
        diff_b = np.abs(medians_frame_b - curr_b).astype(np.uint8)

        kernel = np.ones((4, 4), np.uint8)
        # dilation = cv2.dilate(mask_or, kernel, iterations=1)
        blue_mask = (curr_b < 140).astype(np.uint8)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=2).astype(np.uint8)

        # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        # blue_mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        frame_after_or_after_blue = np.copy(curr)
        frame_after_or_after_blue[:, :, 0] = frame_after_or_after_blue[:, :, 0] * blue_mask
        frame_after_or_after_blue[:, :, 1] = frame_after_or_after_blue[:, :, 1] * blue_mask
        frame_after_or_after_blue[:, :, 2] = frame_after_or_after_blue[:, :, 2] * blue_mask
        out.write(frame_after_or_after_blue)
        blue_mask = 1 - blue_mask
        blue_mask *= 255
        blue_mask_out.write(blue_mask)

    after_or_using_blue_out.release()
    blue_mask_out.release()
    release_video_files(cap, out)
