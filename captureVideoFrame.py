import cv2
from pathlib import Path
import os


inputVideoPath = "/Users/victorialu/Downloads/IMG_2268.MOV"
outputFrameDir = "/Users/victorialu/Downloads/"
frame_to_get_sec = 5
video = cv2.VideoCapture(inputVideoPath)
if video.isOpened():
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_to_get = int(fps*frame_to_get_sec)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_to_get)
    success, frame = video.read()
    if success:
        output_filepath = os.path.join(outputFrameDir, f"{Path(inputVideoPath).stem}_time_{frame_to_get_sec}.jpg")
        cv2.imwrite(output_filepath, frame)
        print(f"Frame has been saved to {output_filepath}")