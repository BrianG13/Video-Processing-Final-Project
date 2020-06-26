from video_stabilization import stabilize_video
from background_substraction import background_substraction
from matting import video_matting
from tracking import track_video
import cv2


stabilize_video('INPUT_SHORT.avi')
background_substraction('stabilize.avi')
video_matting('stabilize.avi','binary.avi',cv2.imread('background.jpg'))
track_video('matted.avi')
