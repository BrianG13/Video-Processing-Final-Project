from video_stabilization import stabilize_video

MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 20
BLOCK_SIZE = 5
SMOOTH_RADIUS = 50

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
stabilize_video('INPUT.avi', output_path, good_features_to_track_params, SMOOTH_RADIUS)
