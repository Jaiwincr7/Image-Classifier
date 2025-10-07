# 🧠 Image Classifier (CIFAR-10 + MobileNetV2)

This Streamlit app compares predictions from a **custom CNN trained on CIFAR-10** and a **pre-trained MobileNetV2** model.

## 🔍 Features
- Upload any image (JPG/PNG)
- Compare CIFAR-10 (10 classes) and MobileNetV2 (1000+ ImageNet classes)
- Automatically downloads model from Google Drive

## 🚀 Deployment
To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
