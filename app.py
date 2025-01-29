import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model("model_resnet4.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define sorted emotion labels
emotion_labels = sorted(["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"])

# Define a function for prediction
def predict_image(image):
    target_size = (224, 224)  # Change target size to match model input shape
    image = cv2.resize(image, target_size)  # Resize to match model input
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = img_array[np.newaxis, ...]  # Add batch dimension to make it (1, 224, 224, 3)
    predictions = model.predict(img_array)  # Get predictions
    predictions = predictions[0]  # Extract the first prediction from the batch
    predicted_label = emotion_labels[np.argmax(predictions)]  # Get label with the highest score
    confidence = np.max(predictions)  # Get the confidence score
    return predicted_label, confidence

# Streamlit app
st.title("Real-Time Emotion Detection")

st.write("Turn on your camera to detect emotions in real time.")

# Add a button to start the camera
if st.button("Start Camera"):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Stream video
    frame_window = st.image([])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image. Make sure your camera is connected.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict emotion
        predicted_label, confidence = predict_image(rgb_frame)

        # Overlay the predicted label on the frame
        cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update the video frame in Streamlit
        frame_window.image(frame, channels="BGR")

    cap.release()
