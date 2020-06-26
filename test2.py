from utils import get_video_files, load_entire_video, write_video

cap, w, h, fps = get_video_files(path='INPUT.avi')

frames_bgr = load_entire_video(cap, color_space='bgr')

write_video(output_path='INPUT_SHORT.avi', frames=frames_bgr[:95], fps=fps, out_size=(w, h), is_color=True)
