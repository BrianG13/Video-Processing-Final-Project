import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import get_video_files, release_video_files, smooth, plot_img_with_points, fixBorder, write_video


def stabilize_video(input_video_path, output_video_path, good_features_to_track, smooth_radius):
    # Read input video
    # cap = cv2.VideoCapture(input_video_path)
    cap, out,w, h, fps = get_video_files(path=input_video_path, output_name=output_video_path, is_color=True)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream


    # Read first frame
    _, prev = cap.read()
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 9), np.float32)
    transforms_list, stabilized_frames_list = [] , []
    for i in range(n_frames):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=good_features_to_track['maxCorners'],
                                           qualityLevel=good_features_to_track['qualityLevel'],
                                           minDistance=good_features_to_track['minDistance'],
                                           blockSize=good_features_to_track['blockSize'])
        # plot_img_with_points(prev_gray, prev_pts)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m, _ = cv2.findHomography(prev_pts, curr_pts)  # will only work with OpenCV-3 or less

        # Store transformation
        transforms[i] = m.flatten()  # [m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2],]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory, smooth_radius)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference
    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # for i in range(transforms_smooth.shape[1]):
    #     plt.plot(transforms_smooth[:, i], label='transforms_smooth ')
    #     plt.plot(transforms[:, i], label='transforms')
    #     plt.legend()
    #     plt.show()

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 1):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Apply affine wrapping to the given frame
        if i == 0:
            frame_stabilized = frame
            out.write(frame_stabilized)

        m = transforms_smooth[i].reshape((3, 3))

        frame_stabilized = cv2.warpPerspective(frame, m, (w, h))

        frame_stabilized = fixBorder(frame_stabilized)
        stabilized_frames_list.append(frame_stabilized)
        transforms_list.append(m)

    release_video_files(cap, out)
    write_video('stabilize.avi', stabilized_frames_list, fps, (w, h), is_color=True)

    return transforms_list