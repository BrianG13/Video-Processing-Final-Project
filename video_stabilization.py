import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import get_video_files, release_video_files, smooth, plot_img_with_points


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def stabilize_video(input_video_path, output_video_path, good_features_to_track, smooth_radius):
    # Read input video
    # cap = cv2.VideoCapture(input_video_path)
    cap, out = get_video_files(input_video_path, output_video_path, isColor=True)
    backSub = cv2.createBackgroundSubtractorKNN()

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cv2.CAP_PROP_FPS)
    # Define the codec for output video
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # Set up output video
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Read first frame
    _, prev = cap.read()
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 9), np.float32)

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
        print(np.sum(err))
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

    # Write n_frames-1 transformed frames
    for i in range(n_frames):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # # Extract transformations from the new transformation array
        # dx = transforms_smooth[i, 0]
        # dy = transforms_smooth[i, 1]
        # da = transforms_smooth[i, 2]
        #
        # # Reconstruct transformation matrix accordingly to new values
        # m = np.zeros((2, 3), np.float32)
        # m[0, 0] = np.cos(da)
        # m[0, 1] = -np.sin(da)
        # m[1, 0] = np.sin(da)
        # m[1, 1] = np.cos(da)
        # m[0, 2] = dx
        # m[1, 2] = dy

        # # Reconstruct transformation matrix accordingly to new values
        # m = np.zeros((2, 3), np.float32)
        # m[0, 0] = transforms_smooth[i, 0]
        # m[0, 1] = transforms_smooth[i, 1]
        # m[1, 0] = transforms_smooth[i, 2]
        # m[1, 1] = transforms_smooth[i, 3]
        # m[0, 2] = transforms_smooth[i, 4]
        # m[1, 2] = transforms_smooth[i, 5]

        # Apply affine wrapping to the given frame
        if i == 0:
            frame_stabilized = frame
        else:
            m = transforms_smooth[i - 1].reshape((3, 3))
            frame_stabilized = cv2.warpPerspective(frame, m, (w, h))


        # Write the frame to the file
        # frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        # if frame_out.shape[1] > 1920:
        #     frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]), int(frame_out.shape[0])));

        # cv2.imshow("Before and After", frame_out)
        # cv2.waitKey(10)
        out.write(frame_stabilized)

    release_video_files(cap, out)
