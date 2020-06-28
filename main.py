from video_stabilization import stabilize_video
from background_subtraction import background_subtraction
from matting import video_matting
from tracking import track_video
import cv2
import logging

LOG_FILENAME = 'RunTimeLog.txt'
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    filename=LOG_FILENAME,
                    filemode='w',
                    level=logging.INFO)

stabilize_video('INPUT.avi')
background_subtraction('stabilize.avi')
video_matting('stabilize.avi','binary.avi',cv2.imread('background.jpg'))
track_video('matted.avi')
