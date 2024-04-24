import cv2
from ultralytics import YOLO
import time
# Load the YOLOv8 model
model = YOLO(r'C:\Users\jozon\Downloads\yolo\v8n\best02.pt')

# Open the video file
video_path = r"D:\pycharm\pythonProject1\utils\video\M_10102022034410_0000000015636581_2_001_0007-01modify_fps_rate.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()


    if success:
        start = time.perf_counter()
        # Run YOLOv8 inference on the frame
        results = model(frame,device='cuda',show_boxes=False,)
        end = time.perf_counter()
        total = end - start
        fps = 1 / (end - start)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame,f"FPS: {int(fps)}",(0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()