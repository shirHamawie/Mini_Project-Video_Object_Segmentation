import cv2
import numpy as np


def extract_first_frame(name, resize_ratio=1.0):
    video_path = f"data/videos/{name}.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to read video file")
    if resize_ratio < 1.0:
        frame = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)
    cap.release()
    save_path = f"data/images/{name}/first_frame_ratio{resize_ratio}.png"
    cv2.imwrite(save_path, frame)


def extract_background(name, resize_ratio=1.0):
    frame_path = f"data/images/{name}/first_frame_ratio{resize_ratio}.png"
    foreground_path = f"data/images/{name}/foreground_ratio{resize_ratio}.png"
    frame = cv2.imread(frame_path)
    foreground = cv2.imread(foreground_path)
    background = np.zeros_like(frame)
    background[foreground == 0] = frame[foreground == 0]
    save_path = f"data/images/{name}/background_ratio{resize_ratio}.png"
    cv2.imwrite(save_path, background)


if __name__ == '__main__':
    name = "swan"
    resize_ratio = 1.0
    # extract_first_frame(name, resize_ratio)
    extract_background(name, resize_ratio)
