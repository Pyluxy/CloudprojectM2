import torch
import glob
import cv2
import numpy as np
import json
import os
from google.cloud import pubsub_v1  # pip install google-cloud-pubsub

# Set up Google Application Credentials
files = glob.glob("*.json")
if files:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]
else:
    raise FileNotFoundError("No JSON key file found for Google Cloud authentication.")

# Project and Pub/Sub configuration
PROJECT_ID = "seismic-kingdom-451318-f5"
TOPIC_NAME = "Imageresults"

# Initialize Pub/Sub publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)
print(f"Publishing messages to {topic_path}...")

# Load Object Detection and Depth Estimation models
YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')
MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
MIDAS_MODEL.to("cuda" if torch.cuda.is_available() else "cpu").eval()

IMAGE_DIR = "images/"
OUTPUT_DIR = "output/"

image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]

for img_path in image_files:
    image = cv2.imread(img_path)
    results = YOLO_MODEL(image)
    detections = results.pred[0]

    boxes = detections[:, :4].cpu().numpy()
    confidence_scores = detections[:, 4].cpu().numpy()
    class_ids = detections[:, 5].cpu().numpy().astype(int)

    detected_objects = []
    for idx, (box, conf, class_id) in enumerate(zip(boxes, confidence_scores, class_ids)):
        if class_id == 0:  # Filter only 'person' class
            x1, y1, x2, y2 = map(int, box)
            depth_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth_frame = cv2.resize(depth_frame, (256, 256))
            depth_tensor = torch.tensor(depth_frame).permute(2, 0, 1).float().unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu") / 255.0
            
            with torch.no_grad():
                depth_map = MIDAS_MODEL(depth_tensor).squeeze().cpu().numpy()
            depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
            estimated_depth = np.median(depth_map_resized[y1:y2, x1:x2])
            
            detected_objects.append({
                "Class": "person",
                "Confidence Level": float(conf),
                "Bounding Box": [x1, y1, x2, y2],
                "Estimated Depth": float(estimated_depth)
            })

    detection_payload = {"image_name": os.path.basename(img_path), "detections": detected_objects}
    try:
        future = publisher.publish(topic_path, json.dumps(detection_payload).encode("utf-8"))
        future.result()
        print(f"Published detection results for {img_path}")
    except Exception as e:
        print(f"Failed to publish results for {img_path}: {e}")
    
    for detection in detected_objects:
        x1, y1, x2, y2 = detection["Bounding Box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person {detection['Confidence Level']:.2f}, Depth: {detection['Estimated Depth']:.2f}m"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(output_path, image)

print("Object detection and processing complete.")
