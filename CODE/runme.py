from video_stabilization import stabilize_video
from background_subtraction import background_subtraction
from matting import video_matting
from tracking import track_video
import cv2
import logging

LOG_FILENAME = '../Outputs/RunTimeLog.txt'
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    filename=LOG_FILENAME,
                    filemode='w',
                    level=logging.INFO)

stabilize_video('../Input/INPUT.avi')
background_subtraction('../Outputs/stabilize.avi')
video_matting('../Outputs/stabilize.avi','../Outputs/binary.avi',cv2.imread('../Input/background.jpg'))
track_video('../Outputs/matted.avi')
