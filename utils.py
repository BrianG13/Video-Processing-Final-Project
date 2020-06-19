import numpy as np
import cv2
from matplotlib import pyplot as plt


def plot_img_with_points(img, points):
    corners = np.int0(points)
    print(corners.shape)
    color = (0, 0, 255)  # color in BGR
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 5, color, -1)
    plt.imshow(img)
    plt.show()


def get_video_files(path, output_name, isColor):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (width, height)
    out = cv2.VideoWriter(output_name, fourcc, fps, out_size, isColor=isColor)
    return cap, out


def release_video_files(cap, out):
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def movingAverage(curve, radius):
    """
    define a moving average filter that takes in any curve ( i.e. a 1-D of numbers) as an input
    and returns the smoothed version of the curve
    :param curve:
    :param radius:
    :return:
    """
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    # plt.plot(curve, label='original curve')
    # plt.plot(curve_smoothed, label='smoothed curve')
    # plt.show()
    return curve_smoothed


def smooth(trajectory, smooth_radius):
    """
    takes in the trajectory and performs smoothing on the three components
    :param smooth_radius:
    :param trajectory:
    :return:
    """
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(9):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=smooth_radius)

    return smoothed_trajectory
