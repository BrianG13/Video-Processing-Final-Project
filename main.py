import cv2
from video_stabilization import stabilize_video
from background_substraction import background_substraction
from matting import video_matting

MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 30
BLOCK_SIZE = 3
SMOOTH_RADIUS = 5

good_features_to_track_params = {
    'maxCorners': MAX_CORNERS,
    'qualityLevel': QUALITY_LEVEL,
    'minDistance': MIN_DISTANCE,
    'blockSize': BLOCK_SIZE
}

folder_name = 'brian_experiment'
output_video_name = f'Stabilized_video_maxCorners{MAX_CORNERS}_qualityLevel{QUALITY_LEVEL}_minDistance{MIN_DISTANCE}' \
                    f'_blockSize{BLOCK_SIZE}_radius{SMOOTH_RADIUS}.avi'
output_path = f'{folder_name}/{output_video_name}'
# stabilize_video('INPUT.avi', output_path, good_features_to_track_params, SMOOTH_RADIUS)
# background_substraction('stabilized_video.avi','black_and_white.avi')
video_matting('stabilized_video.avi','background_substraction_mask.avi',
              'output_matting.avi',cv2.imread('background.jpg'))
