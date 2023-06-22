import cv2


def resize_video(name, resize_ratio=2):
    cap = cv2.VideoCapture(f"data/new_videos/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_ratio)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_ratio)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"data/new_videos/{name}_resized.mp4", fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    name = "swan_best"
    resize_ratio = 2
    resize_video(name, resize_ratio)
