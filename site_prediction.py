import os
import streamlit as st
# import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import gdown

# ---------------------- CONFIG ----------------------
MODEL_PATH = "cnn_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=10Se1OUFJKhbT2U11kgU-rx-NAAeFVmp0"
  # Replace with your direct download link
# ----------------------------------------------------

def download_cnn_model():
    """Download the CNN model from Google Drive if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading CNN model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

def load_cifar_model():
    download_cnn_model()
    cnn = tf.keras.models.load_model(MODEL_PATH)
    return cnn

def load_mobilenet():
    model = MobileNetV2(weights='imagenet')
    return model

# ---------------------- PREPROCESSING ----------------------
def preprocess_cifar(img_file):
    img = img_file.resize((32, 32))  # Resize using PIL
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def preprocess_mobilenet(img_file):
    img = img_file.resize((224, 224))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# ---------------------- CLASSIFICATION ----------------------
def classify(cnn, mobilenet, img_file):
    try:
        # CNN prediction
        img_cifar = preprocess_cifar(img_file)
        result_cnn = cnn.predict(img_cifar)

        # MobileNet prediction
        img_mobilenet = preprocess_mobilenet(img_file)
        result_mobilenet = mobilenet.predict(img_mobilenet)
        decoded_predictions = decode_predictions(result_mobilenet, top=3)[0]

        # CIFAR-10 class names
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        top_3_indices = np.argsort(result_cnn[0])[-3:][::-1]
        cifar_predictions = [(class_names[idx], float(result_cnn[0][idx])) for idx in top_3_indices]

        return cifar_predictions, decoded_predictions

    except Exception as e:
        st.error(f"Error Classifying the image: {str(e)}")
        return None, None

# ---------------------- STREAMLIT APP ----------------------
def main():
    st.set_page_config(page_title="Image Classifier", layout='wide')
    st.title('🧠 Image Classifier (CIFAR-10 + MobileNetV2)')

    # Display model info
    data = {
        'Model': ['Custom CNN', 'MobileNetV2'],
        'Definitions': [
            'A CNN trained from scratch on the CIFAR-10 dataset (10 classes).',
            'MobileNetV2 is a pre-trained ImageNet model optimized for mobile/embedded use.'
        ],
        'Working': [
            'CNN extracts features using convolutional layers and classifies images with dense layers.',
            'MobileNetV2 uses pre-trained weights with inverted residual blocks for efficient inference.'
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)
    st.write('Upload an image to classify it using both models.')

    @st.cache_resource
    def load_models():
        return load_cifar_model(), load_mobilenet()

    cnn_model, mobilenet_model = load_models()

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            btn = st.button('Classify Image')

        with col2:
            if btn:
                with st.spinner("Analyzing Image..."):
                    img = Image.open(uploaded_file)
                    cifar_preds, mobilenet_preds = classify(cnn_model, mobilenet_model, img)

                    if cifar_preds:
                        st.subheader("CIFAR-10 Predictions (Custom CNN)")
                        for cls, prob in cifar_preds:
                            st.write(f"**{cls}**: {prob:.4f}")

                    if mobilenet_preds:
                        st.subheader("MobileNetV2 Predictions (ImageNet)")
                        for name, desc, score in mobilenet_preds:
                            st.write(f"**{desc}**: {score:.4f}")

if __name__ == "__main__":
    main()

