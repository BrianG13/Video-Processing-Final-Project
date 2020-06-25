
import numpy as np
import numpy.matlib

import cv2

from tracking_utils import predictParticles, compNormHist, measure, sampleParticles, showParticles, build_KDE, \
    compute_KL_div
from utils import get_video_files, load_entire_video, write_video


def track_video(input_video_path):
    cap_stabilize, _, W, H, fps = get_video_files(input_video_path, 'delete.avi', isColor=True)
    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')

    initBB = cv2.selectROI("Frame", frames_bgr[0], fromCenter=False,
                                                  showCrosshair=True)

    x, y, w, h = initBB
    if any([x==0, y==0, w==0, h==0]):
        x, y, w, h = 180, 60, 340, 740  # simply hardcoded the values
    track_window = (x, y, w, h)
    
    # set up the ROI for tracking
    roi = frames_bgr[0][y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0,1], mask, [256,256], [0, 256,0,256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0)
    ret = True
    tracking_frames_list = [cv2.rectangle(frames_bgr[0], (x, y), (x + w, y + h), (0,255,0), 2)]
    for frame in frames_bgr[1:]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0,1], roi_hist, [0, 256,0,256], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        tracked_img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        tracking_frames_list.append(tracked_img)

    write_video('OUTPUT.avi', tracking_frames_list, fps, (W,H), is_color=True)



def track_custom_particles(input_video_path):
    cap_stabilize, _, W, H, fps = get_video_files(input_video_path, 'delete.avi', isColor=True)
    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')

    # SET NUMBER OF PARTICLES
    N = 100

    # Initial Settings

    s_initial = [230,  # x center
                 550,  # y center
                 170,  # half width
                 370,  # half height
                 0,  # velocity x
                 0]  # velocity y

    # CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
    S = predictParticles(np.matlib.repmat(s_initial, N, 1).T)

    # LOAD FIRST IMAGE
    I = frames_bgr[0]

    # COMPUTE NORMALIZED HISTOGRAM
    # q = compNormHist(I, s_initial)
    #
    q = build_KDE(I,s_initial)
    # result = compute_KL_div(initial_pdf,initial_pdf)

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


def track_using_cv2_trackers(input_video_path):
    cap_stabilize, _, W, H, fps = get_video_files(input_video_path, 'delete.avi', isColor=True)
    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')

    (major, minor) = cv2.__version__.split(".")[:2]
    print(major)
    print(minor)
    tracker = cv2.TrackerMIL_create()
    # initialize the bounding box coordinates of the object we are going to track



    # show the output frame
    # cv2.imshow("Frame", frames_bgr[0])
    # cv2.waitKey(0)
    # select the bounding box of the object we want to track (make
    # sure you press ENTER or SPACE after selecting the ROI)
    # initBB = cv2.selectROI("Frame", frames_bgr[0], fromCenter=False,
    #                        showCrosshair=True)

    # start OpenCV object tracker using the supplied bounding box
    # coordinates, then start the FPS throughput estimator as well
    initBB = (76, 248, 296, 695)
    tracker.init(frames_bgr[0], initBB)


    i=0
    for frame in frames_bgr[1:]:
        i+=1
        print(i)
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
        cv2.imshow(f'{i}',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == "q":
            exit()
