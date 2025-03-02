from google.cloud import pubsub_v1  # pip install google-cloud-pubsub
import glob  # For searching for JSON files
import json
import os

# Set up Google Application Credentials
files = glob.glob("*.json")
if files:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0]
else:
    raise FileNotFoundError("No JSON key file found for Google Cloud authentication.")

# Project and Pub/Sub configuration
PROJECT_ID = "seismic-kingdom-451318-f5"
TOPIC_NAME = "Imageresults"
SUBSCRIPTION_ID = "Imageresults-sub"

# Initialize Pub/Sub subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

print(f"Listening for messages on {subscription_path}...")

# Callback function to handle received messages
def process_message(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        message_data = message.data.decode("utf-8")
        detection_info = json.loads(message_data)
        print(f"Processing image: {detection_info['image_name']}")

        for detection in detection_info['detections']:
            print(f"Detected: {detection['Class']}, Confidence: {detection['Confidence Level']:.2f}, Depth: {detection['Estimated Depth']:.2f}m")
        
        message.ack()  # Acknowledge successful message processing
    except Exception as e:
        print(f"Error processing message: {e}")

# Subscribe and listen for messages
with subscriber:
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)
    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()

print("Message consumption complete.")
