🌿 GreenClassify
Deep Learning-Based Approach for Vegetable Image Classification

### 📌 Project Overview

GreenClassify is a deep learning–based web application designed to accurately identify and classify different types of vegetables from images.
The project leverages Convolutional Neural Networks (CNNs) with transfer learning to provide fast and reliable vegetable classification through a user-friendly web interface built using Flask.

This project is developed as part of SkillWallet / SmartInternz Internship Program under the category Deep Learning.

# 🎯 Project Objectives

To build an automated vegetable image classification system

To apply transfer learning using a pre-trained CNN model

To reduce manual effort in vegetable identification

To deploy the trained model as a Flask web application

To provide an interactive and responsive UI for end users

## 🧠 CNN Model Used

Model: MobileNetV2

Approach: Transfer Learning

Pre-trained On: ImageNet

Input Size: 224 × 224 × 3

Why MobileNetV2?

Lightweight and efficient architecture

Faster inference time

Suitable for real-time and web applications

Good balance between accuracy and performance

## 🏗️ Model Architecture
Input Image (224x224x3)
        ↓
MobileNetV2 (Pre-trained CNN Backbone)
        ↓
Global Average Pooling
        ↓
Dense Layer (Softmax)
        ↓
Predicted Vegetable Class

## 🛠️ Technologies Used
Programming & Frameworks
Python
TensorFlow / Keras
Flask
Machine Learning
Convolutional Neural Networks (CNN)
Transfer Learning
Image Classification
Frontend
HTML5
CSS3
JavaScript
Tools & Platforms
Kaggle (Model Training)
VS Code
Git & GitHub

## 📂 Project Structure
VEGETABLE_CLASSIFICATION/
│
├── dataset/
│   ├── train/
│   ├── test/
│   └── validation/
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── uploads/
│   └── background.jpg
│
├── templates/
│   └── index.html
│
├── app.py
├── vegetable_classifier_model.h5
├── README.md
└── requirements.txt


### 🔄 Project Workflow

Data Collection
Vegetable images collected and organized by class
Data Pre-Processing
Image resizing
Normalization
Data augmentation
Model Building
MobileNetV2 as base model
Custom classification head added
Model Training
Adam optimizer
Categorical Cross-Entropy loss
Early stopping for better generalization
Model Evaluation
Validation accuracy monitoring
Web Application Development
Flask backend
Responsive UI
Prediction
Upload vegetable image
Model predicts vegetable class

## 📊 Scenarios & Use Cases
🥕 Automated Vegetable Sorting

Helps processing facilities automatically classify vegetables in bulk shipments.

🛒 Retail & Inventory Management

Assists retailers in identifying vegetables for pricing and inventory tracking.

🌾 Agricultural Support

Useful for farmers and agri-tech platforms to identify crops quickly.

### 🚀 How to Run the Project Locally
1️⃣ Clone the Repository
git clone "https://github.com/Shree-2516/GreenClassify.git"
cd VEGETABLE_CLASSIFICATION
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
python app.py
5️⃣ Open in Browser
http://127.0.0.1:5000

## 🖼️ Application Features

Single-page scrollable website

Attractive and responsive UI

Image upload with preview

Real-time vegetable prediction

Hover, glow, and animation effects

Smooth user experience

## 📌 Deliverables

✅ Trained CNN Model (.h5)

✅ Flask Web Application

✅ Project Documentation

✅ Source Code

✅ Demo Interface

## 👨‍💻 Author

Project Name: Shreeyash
Domain: Deep Learning
Internship Platform: SkillWallet / SmartInternz

## 📜 License

This project is developed for educational and internship purposes only.