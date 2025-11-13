import cv2
import gradio as gr
import numpy as np


def vid_inf(vid_path, contour_thresh):
    contour_thresh = int(contour_thresh)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error opening video file")
        yield None, None
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = "output_recorded.mp4"
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = backSub.apply(frame)

        _, mask_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_cleaned = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > contour_thresh]

        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)

        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        out.write(frame_out)

        if count % 12 == 0:
            yield frame_rgb, None
        count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    yield None, output_video


# Gradio interface
input_video = gr.Video(label="Input Video")
contour_thresh = gr.Slider(
    0,
    10000,
    value=1000,
    label="Contour Threshold",
    info="Set the minimum size of moving objects to detect (in pixels).",
)
output_frames = gr.Image(label="Output Frames")
output_video_file = gr.Video(label="Output video")

app = gr.Interface(
    theme="gstaff/xkcd",
    fn=vid_inf,
    inputs=[input_video, contour_thresh],
    outputs=[output_frames, output_video_file],
    title="Motion Detection using OpenCV",
    description="A Gradio app that uses background subtraction and contour detection to highlight moving objects in a video.",
    flagging_mode="never",
    examples=[["./sample/car.mp4", 1000], ["./sample/motion_test.mp4", 5000], ["./sample/home.mp4", 4500]],
    cache_examples=False,
)

app.queue().launch(pwa=True)
