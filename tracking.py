
import numpy as np
import numpy.matlib

import cv2

from tracking_utils import predictParticles, compNormHist, measure, sampleParticles, showParticles
from utils import get_video_files, load_entire_video


def track_video(input_video_path):
    cap_stabilize, _, W, H, fps = get_video_files(input_video_path, 'delete.avi', isColor=True)
    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')

    # SET NUMBER OF PARTICLES
    N = 100

    # Initial Settings
    s_initial = [230,  # x center // TODO
                 550,  # y center
                 170,  # half width
                 370,  # half height
                 0,  # velocity x
                 0]  # velocity y

    # CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
    a = np.matlib.repmat(s_initial, N, 1).T

    S = predictParticles(np.matlib.repmat(s_initial, N, 1).T)

    # LOAD FIRST IMAGE
    I = frames_bgr[0]

    # COMPUTE NORMALIZED HISTOGRAM
    q = compNormHist(I, s_initial)
    #
    C, _ = measure(I, S, q)
    #
    # s_t_tag = sampleParticles(S, c)

    images_processed = 1

    # MAIN TRACKING LOOP
    for frame in frames_bgr[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        I = frame

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sampleParticles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predictParticles(S_next_tag)

        C, W = measure(I, S, q)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if images_processed % 1 == 0:
            showParticles(I, S, W, images_processed, "")


def opencv_contrib_track(input_video_path):
    cap_stabilize, _, W, H, fps = get_video_files(input_video_path, 'delete.avi', isColor=True)
    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')

    (major, minor) = cv2.__version__.split(".")[:2]

    tracker = cv2.TrackerCSRT_create()
    # initialize the bounding box coordinates of the object we are going to track



    # show the output frame
    # cv2.imshow("Frame", frames_bgr[0])
    # cv2.waitKey(0)
    # select the bounding box of the object we want to track (make
    # sure you press ENTER or SPACE after selecting the ROI)
    initBB = cv2.selectROI("Frame", frames_bgr[0], fromCenter=False,
                           showCrosshair=True)
    # start OpenCV object tracker using the supplied bounding box
    # coordinates, then start the FPS throughput estimator as well
    tracker.init(frames_bgr[0], initBB)



    for frame in frames_bgr[1:]:
        # check to see if we are currently tracking an object
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", 'KCF'),
            ("Success", "Yes" if success else "No"),
            ("FPS", f"{fps}"),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)