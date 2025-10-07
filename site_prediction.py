import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

def load_cifar_model():
    cnn = tf.keras.models.load_model("cnn_model.h5")
    return cnn

def load_mobilenet():
    model = MobileNetV2(weights='imagenet')
    return model

# different preprocessing method for different models
def preprocess_cifar(img_file):
    img = np.array(img_file)
    img = cv2.resize(img, (32, 32))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def preprocess_mobilenet(img_file):
    img = np.array(img_file)
    img = cv2.resize(img, (224, 224))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def classify(cnn, mobilenet, img_file):
    try:
        # cnn prediction
        img_cifar = preprocess_cifar(img_file)
        result_cnn = cnn.predict(img_cifar)

        # mobilenet prediction
        img_mobilenet = preprocess_mobilenet(img_file)
        result_mobilenet = mobilenet.predict(img_mobilenet)
        decoded_predictions = decode_predictions(result_mobilenet, top=3)[0]

        # class in the cifar based dataset
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

def main():
    st.set_page_config(page_title="Image Classifier", layout='wide')
    st.title('🧠 Image Classifier (CIFAR-10 + MobileNetV2)')

    data = {
        'Model': ['Custom CNN', 'MobileNetV2'],
        'Definitions': [
            'A Convolutional Neural Network (CNN) is a deep learning architecture that automatically learns spatial hierarchies of features from images. This model was trained from scratch on the CIFAR-10 dataset to classify 10 categories such as airplanes, dogs, and ships.',
            'MobileNetV2 is a pre-trained convolutional neural network designed for high accuracy with minimal computation. It uses depthwise separable convolutions and is optimized for mobile and embedded vision applications.'
        ],
        'Working': [
            'Our CNN extracts features using multiple convolutional and pooling layers, then classifies images through dense layers. It learns all parameters directly from the CIFAR-10 dataset during training.',
            'MobileNetV2 uses pre-trained ImageNet weights and relies on inverted residual blocks with linear bottlenecks to achieve efficient inference while maintaining accuracy. It’s used here to compare transfer learning performance with the custom CNN.'
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)

    st.write('Upload an image to classify it using both a custom CNN and MobileNetV2.')

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
