import cv2
from video_stabilization import stabilize_video
from background_substraction import background_substraction
from matting import video_matting
from tracking import track_video



stabilize_video('INPUT.avi')
# background_substraction('stabilized_video.avi')
# video_matting('stabilized_video.avi','background_substraction_mask.avi',cv2.imread('background.jpg'))
# track_video('matted_video.avi')