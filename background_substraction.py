import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import get_video_files, release_video_files, smooth, plot_img_with_points


# from Faster_Kmeans_master.Code.kmeans import Kmeans

def background_substraction(input_video_path, output_video_path):
    # Read input video
    # cap = cv2.VideoCapture(input_video_path)
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

    u_results = []  ## OPTICAL FLOW
    v_results = []  ## OPTICAL FLOW
    frames_bgr = [prev]  # MEDIAN TRY
    frames_hsv = [cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)]
    for i in range(n_frames):
        print("Frame: " + str(i) + "/" + str(n_frames))
        # Detect feature points in previous frame
        # if i == 40:
        #     cv2.imwrite("frame40.png", curr)
        #     print('hi')

        # Read next frame
        success, curr = cap.read()
        if not success:
            break
        frames_bgr.append(curr)
        frames_hsv.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))

        # Convert to grayscale
        # curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # for every frame,
        # get current_frame_gray

        '''Optical flow try'''
        # farneback_params = {
        #     'pyr_scale': 0.5,
        #     'levels': 3,
        #     'winsize': 15,
        #     'iterations': 3,
        #     'poly_n': 5,
        #     'poly_sigma': 1.2,
        #     'flags': cv2.OPTFLOW_USE_INITIAL_FLOW
        # }
        # flow = np.zeros((h, w, 2), dtype=np.float32)
        # flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow, **farneback_params)
        # u_results.append(flow[:,:,0])
        # v_results.append(flow[:, :, 1])

        # scaled_u = 255 * (flow[:, :, 0] - np.min(flow[:, :, 0])) / np.ptp(flow[:, :, 0])
        # scaled_u = np.uint8(scaled_u)
        # u_results.append(scaled_u)
        # scaled_v = 255 * (flow[:, :, 1] - np.min(flow[:, :, 1])) / np.ptp(flow[:, :, 1])
        # scaled_v = np.uint8(scaled_v)
        # v_results.append(scaled_v)
        # cv2.imshow('ImageWindow', u_matrix)
        # cv2.waitKey(0)
        '''Optical flow try - END'''

        '''K-means try'''
        # pointList = flow.reshape((h*w,2))
        # pointList2 = pointList[::100,:]
        # small = cv2.resize(prev_gray, (0, 0), fx=1/10, fy=1/10)
        # cv2.imshow('small',small)
        # cv2.waitKey(0)
        # kmeans_result = Kmeans(2, pointList, 100)
        '''K-means try - END'''

        # prev_gray = curr_gray
        continue

    frames_bgr = np.asarray(frames_bgr)
    frames_hsv = np.asarray(frames_hsv)
    medians_frame_bgr = np.median(frames_bgr, axis=0)
    medians_frame_b, medians_frame_g, medians_frame_r = cv2.split(medians_frame_bgr)
    medians_frame_hsv = np.median(frames_hsv, axis=0)
    medians_frame_h, medians_frame_s, medians_frame_v = cv2.split(medians_frame_hsv)

    out_size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    # h_out = cv2.VideoWriter('h_results.avi', fourcc, fps, out_size, isColor=False)
    s_out = cv2.VideoWriter('s_results.avi', fourcc, fps, out_size, isColor=False)
    v_out = cv2.VideoWriter('v_results.avi', fourcc, fps, out_size, isColor=False)
    b_out = cv2.VideoWriter('b_results.avi', fourcc, fps, out_size, isColor=False)
    # g_out = cv2.VideoWriter('g_results.avi', fourcc, fps, out_size, isColor=False)
    # r_out = cv2.VideoWriter('r_results.avi', fourcc, fps, out_size, isColor=False)
    # mask_s_out = cv2.VideoWriter('mask_s.avi', fourcc, fps, out_size, isColor=False)
    # mask_v_out = cv2.VideoWriter('mask_v.avi', fourcc, fps, out_size, isColor=False)
    mask_weighted_out = cv2.VideoWriter('mask_weighted.avi', fourcc, fps, out_size, isColor=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        curr_hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
        curr_h, curr_s, curr_v = cv2.split(curr_hsv)
        curr_b, curr_g, curr_r = cv2.split(curr)
        # diff_h = np.abs(medians_frame_h-curr_h).astype(np.uint8)
        diff_s = np.abs(medians_frame_s-curr_s).astype(np.uint8)
        diff_v = np.abs(medians_frame_v-curr_v).astype(np.uint8)
        diff_b = np.abs(medians_frame_b-curr_b).astype(np.uint8)
        # diff_g = np.abs(medians_frame_g-curr_g).astype(np.uint8)
        # diff_r = np.abs(medians_frame_r-curr_r).astype(np.uint8)
        # mask_s = (diff_s > np.mean(diff_s)*5).astype(np.uint8) * 255
        # mask_v = (diff_v > np.mean(diff_v)*7).astype(np.uint8) * 255
        weighted_mask = (diff_s - np.mean(diff_s) * 5 + 0.2*(diff_v - np.mean(diff_v) * 7) > 0).astype(np.uint8) * 255

        # h_out.write(diff_h)
        s_out.write(diff_s)
        v_out.write(diff_v)
        b_out.write(diff_b)
        # g_out.write(diff_g)
        # r_out.write(diff_r)
        # mask_s_out.write(mask_s)
        # mask_v_out.write(mask_v)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 50)
        fontScale = 3
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(weighted_mask, str(i),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('s',weighted_mask)
        cv2.waitKey(0)
        mask_weighted_out.write(weighted_mask)

        # print(f'mean of s: {np.mean(diff_s)}')

    # h_out.release()
    s_out.release()
    v_out.release()
    b_out.release()
    # g_out.release()
    # r_out.release()
    # mask_s_out.release()
    # mask_v_out.release()
    mask_weighted_out.release()

    '''OPTICAL FLOW TRY'''
    # out_size = (w, h)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # v_out = cv2.VideoWriter('v_results.avi', fourcc, fps, out_size, isColor=False)
    # u_out = cv2.VideoWriter('u_results.avi', fourcc, fps, out_size, isColor=False)
    #
    # for u_frame in u_results:
    #     u_out.write(u_frame)
    # u_out.release()
    # for v_frame in v_results:
    #     v_out.write(v_frame)
    # v_out.release()
    '''OPTICAL FLOW TRY - end'''

    release_video_files(cap, out)
