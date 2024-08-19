from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os

from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)

cap = cv2.VideoCapture(0)
screen_size_inch = 14
aspect_ratio = (16, 9)

diagonal_size_cm = np.sqrt(screen_size_inch**2 + aspect_ratio[0]**2)
frame_width = int((aspect_ratio[0] / np.sqrt(aspect_ratio[0]**2 + aspect_ratio[1]**2)) * diagonal_size_cm * 37.8)
frame_height = int(frame_width * aspect_ratio[1] / aspect_ratio[0])

model = YOLO(r"C:\Users\hp\Desktop\flask2\best.pt")
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
tracker = sv.ByteTrack()

red_zone_width = int(frame_width * 0.5)
red_zone_height = int(frame_height * 0.5)

red_zone_x = (frame_width - red_zone_width) // 2
red_zone_y = (frame_height - red_zone_height) // 2

zone_polygon = np.array([
    [red_zone_x, red_zone_y],
    [red_zone_x + red_zone_width, red_zone_y],
    [red_zone_x + red_zone_width, red_zone_y + red_zone_height],
    [red_zone_x, red_zone_y + red_zone_height]
])

zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.red(),
    thickness=2,
    text_thickness=4,
    text_scale=2
)

save_interval_minutes = 1  # Save every 1 minute
last_save_time = datetime.now()

output_directory = "frames_output"
os.makedirs(output_directory, exist_ok=True)


def generate_frames(camera, save_interval_minutes=1, output_directory="frames_output"):
    model = YOLO(r"C:\Users\hp\Desktop\flask2\best.pt")
    last_save_time = datetime.now()

    while True:
        ret, frame = camera.read()
        results = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Filter detections for class_id == 3
        filtered_detections = detections[detections.class_id == 0]

        filtered_labels = [
            f"#{tracker_id}{results.names[class_id]}"
            for class_id, tracker_id in zip(filtered_detections.class_id, filtered_detections.tracker_id)
        ]

        detections = detections[detections.confidence > 0.2]
        labels = [
            f"#{tracker_id}{results.names[class_id]}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        # Temporarily remove the zone from annotated frame
        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        zone.trigger(detections=detections)

        # Check if it's time to save a new frame
        current_time = datetime.now()
        if (current_time - last_save_time).total_seconds() >= save_interval_minutes * 60:
            save_frame(annotated_frame, current_time)
            save_cropped_images(frame, filtered_detections, current_time)

            # Create a DataFrame for zone information
            zone_data = {
                "Class Label": [results.names[class_id] for class_id in detections.class_id],
                "Tracker ID": [int(tracker_id) for tracker_id in detections.tracker_id],
                "Timestamp": [current_time.strftime('%Y-%m-%d %H:%M:%S')] * len(detections)
            }
            zone_df = pd.DataFrame(zone_data)

            # Save the DataFrame to an Excel file
            excel_filename = os.path.join(output_directory, f"zone_info_{current_time.strftime('%Y%m%d%H%M%S')}.xlsx")
            zone_df.to_excel(excel_filename, index=False)

            last_save_time = current_time

        # Add the zone back for video feed
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)

        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        data = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n')


def save_frame(frame, current_time):
    filename = os.path.join(output_directory, f"frame_{current_time.strftime('%Y%m%d%H%M%S')}.jpg")
    cv2.imwrite(filename, frame)


def save_cropped_images(frames, detections, current_time):
    for idx, detection in enumerate(detections.xyxy):
        X1, Y1, X2, Y2 = map(int, detection[:4])
        cropped_images = frames[Y1:Y2, X1:X2]

        # Convert to grayscale
        cropped_images_gray = cv2.cvtColor(cropped_images, cv2.COLOR_BGR2GRAY)

        # Save the cropped grayscale image
        filename = os.path.join(output_directory,
                                f"weight_machine_{current_time.strftime('%Y%m%d%H%M%S')}_{idx}.jpg")
        cv2.imwrite(filename, cropped_images_gray)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    camera_index = int(request.args.get('camera', default='0'))
    cap = cv2.VideoCapture(camera_index)
    return Response(generate_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
