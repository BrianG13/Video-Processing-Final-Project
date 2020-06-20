import cv2
import numpy as np

from utils import (
    get_video_files,
    release_video_files,
    write_video,
    scale_matrix_0_to_255)

from Faster_Kmeans_master.Code.kmeans import Kmeans


def background_substraction(input_video_path, output_video_path):
    # Read input video
    # cap = cv2.VideoCapture(input_video_path)
    cap, out, w, h, fps = get_video_files(input_video_path, output_video_path, isColor=True)
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    _, prev = cap.read()
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Pre-define transformation-store array

    u_results = []
    v_results = []
    for i in range(n_frames):
        print("Frame: " + str(i) + "/" + str(n_frames))
        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': cv2.OPTFLOW_USE_INITIAL_FLOW
        }
        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow, **farneback_params)

        scaled_u = scale_matrix_0_to_255(flow[:, :, 0])
        u_results.append(scaled_u)
        scaled_v = scale_matrix_0_to_255(flow[:, :, 1])
        v_results.append(scaled_v)

        '''K-means try'''
        pointList = flow.reshape((h * w, 2))
        pointList2 = pointList[::100, :]
        kmeans_result = Kmeans(2, pointList2, 100)
        '''K-means try - END'''

        prev_gray = curr_gray
        continue

    write_video('u_opflow_results.avi', frames=u_results, fps=fps, out_size=(w, h), is_color=False)
    write_video('v_opflow_results.avi', frames=v_results, fps=fps, out_size=(w, h), is_color=False)

    release_video_files(cap, out)
