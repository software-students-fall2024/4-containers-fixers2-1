"""
This module contains functions for emotion detection 
using a pre-trained machine learning model.
"""

from flask import Flask, request, jsonify, flash
from datetime import datetime, timezone
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import os

# pylint: disable=all

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "test_secret_key")

# Connect to MongoDB
client = MongoClient("mongodb://mongo:27017/")  # Use Docker service name for MongoDB
db = client["emotion_db"]
emotion_data_collection = db["emotion_data"]

# Load the pre-trained emotion detection model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "face_model.h5")  # pylint: disable=no-member

model = load_model(model_path)  # pylint: disable=no-member

# class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define a dictionary to map model output to emotion text
emotion_dict = {
    0: ("Angry 😡", "Take deep breaths or go for a calming walk."),
    1: ("Disgusted 🥴", "Try to step away and focus on something you enjoy."),
    2: ("Fear 😨", "Remind yourself of your strengths and seek support if needed."),
    3: ("Happy 😊", "Celebrate your happiness by sharing it with someone!"),
    4: ("Sad 😢", "Listen to uplifting music or talk to a trusted friend."),
    5: ("Surprised 😮", "Take a moment to reflect and process the surprise."),
    6: ("Neutral 😐", "Use this balance to focus on tasks or mindfulness."),
}


def save_emotion(emotion):
    try:
        emotion_add = {
            "emotion": emotion,
            "timestamp": datetime.now(datetime.timezone.utc),
        }
        emotion_data_collection.insert_one(emotion_add)
    except Exception as error:
        flash(f"Error saving emotion to database")


@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    """
    Detect emotion from an image sent via POST request and save it to MongoDB.
    """
    try:
        # Check if an image is provided in the request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        if model is None:
            return jsonify({"error": "Emotion detection model not loaded"}), 500

        # Read the image file from the request
        file = request.files["image"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        # Handle case where no faces are detected
        if len(faces) == 0:
            return jsonify({"error": "No faces detected in the image."}), 400

        for x, y, w, h in faces:

            face_roi = frame[y : y + h, x : x + w]
            face_image = cv2.resize(face_roi, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = np.vstack([face_image])

            predictions = model.predict(face_image)
            emotion_index = np.argmax(predictions)
            emotion_label, recommendation = emotion_dict.get(
                emotion_index, ("Unknown", "No recommendation available.")
            )

            try:
                emotion_data_collection.insert_one(
                    {"emotion": emotion_label, "timestamp": datetime.utcnow()}
                )
            except Exception as db_error:
                return (
                    jsonify({"error": f"Database insertion failed: {str(db_error)}"}),
                    500,
                )

            return jsonify({"emotion": emotion_label, "recommendation": recommendation})

    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500

    # Fallback for unexpected cases
    return jsonify({"error": "Unexpected error occurred"}), 500


def run_emotion_detection():
    """
    Opens the camera and runs the emotion detection model in real-time.
    Each frame's detected emotion is displayed and saved to MongoDB.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera (0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        cap.release()
        exit(1)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect emotion from the current frame
        emotion_text = detect_emotion(frame)

        # Display the emotion text on the frame
        cv2.putText(
            frame,
            f"Emotion: {emotion_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Show the frame with the detected emotion
        cv2.imshow("Emotion Detection", frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
