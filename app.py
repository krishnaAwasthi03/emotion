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
st.set_page_config(page_title="Emotion Detection", page_icon="😃", layout="centered")

st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #333;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
        }
    </style>
    <div class="main">
        <p class="title">Emotion Detection from Image</p>
        <p class="subtitle">Upload an image to detect emotions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to numpy array
    image = np.array(image)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
    
    # Predict emotion
    predicted_label, confidence = predict_image(image)
    
    # Display results
    st.markdown(f"<h3 style='text-align: center; color: #444;'>Predicted Emotion: {predicted_label}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: #777;'>Confidence: {confidence:.2f}</h4>", unsafe_allow_html=True)
