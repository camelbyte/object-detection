import argparse
import cv2
import torch
import os

def run_detection_on_image(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return
    # Run inference
    results = model(image)
    results.render()  # draw boxes on image in-place
    output_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(output_path, results.imgs[0])
    print(f"Detection complete. Output saved as {output_path}")
    cv2.imshow("Detection", results.imgs[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_detection_on_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file or stream.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.render()
        cv2.imshow("Detection", results.imgs[0])
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_detection_on_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.render()
        cv2.imshow("Detection", results.imgs[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Basic object detection on an image, video, or webcam using YOLOv5"
    )
    parser.add_argument(
        '--source',
        type=str,
        default="webcam",
        help="Path to image/video file or 'webcam' to use the live camera feed."
    )
    args = parser.parse_args()

    # Load YOLOv5 model from PyTorch Hub (downloads if needed)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()  # set in evaluation mode

    source = args.source.lower()
    if source == "webcam":
        run_detection_on_webcam(model)
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            run_detection_on_image(source, model)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            run_detection_on_video(source, model)
        else:
            print("Unsupported file type. Use an image (jpg/png) or video (mp4/avi/mov/mkv).")
    else:
        print("Source not recognized. Please use 'webcam' or provide a valid file path.")

if __name__ == '__main__':
    main()

