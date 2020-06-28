from video_stabilization import stabilize_video
from bs_new_approach import background_subtraction  # TODO - DELETE
from matting import video_matting
from tracking import track_video
import cv2
import logging

LOG_FILENAME = 'RunTimeLog.txt'
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filename=LOG_FILENAME,
                    filemode='w',
                    level=logging.INFO)

my_logger = logging.getLogger('MyLogger')

exit()

stabilize_video('INPUT.avi')
background_subtraction('stabilize.avi')
video_matting('stabilize.avi','binary.avi',cv2.imread('background.jpg'))
track_video('matted.avi')
