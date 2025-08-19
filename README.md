# DigitVision
# 🖊️ Handwritten Digit Recognition on MNIST Dataset

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)](https://scikit-learn.org/)

## 📌 Project Description
This project focuses on building a **machine learning model** to automatically recognize handwritten digits (0–9) using the widely-used **MNIST dataset**. The MNIST dataset contains **70,000 grayscale images** of handwritten digits, each of size 28x28 pixels.

The project involves:  
- **Data Preprocessing:** Normalizing pixel values, reshaping images, and splitting the dataset into training and testing sets.  
- **Model Development:** Implementing and training models such as **Logistic Regression, Support Vector Machines (SVM), and Convolutional Neural Networks (CNN)** to achieve high accuracy.  
- **Evaluation:** Measuring model performance using **accuracy, precision, recall**, and **confusion matrix**.  
- **Visualization:** Displaying sample predictions and analyzing misclassified images to understand model behavior.

This project demonstrates fundamental concepts in **machine learning** and **deep learning**, including image preprocessing, model training, and evaluation, and provides hands-on experience in solving real-world **computer vision problems**.

## 🛠️ Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras  

## 🚀 Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
Navigate to the project folder:

cd handwritten-digit-recognition


Install required dependencies:

pip install -r requirements.txt

🧠 Usage

Open the Jupyter Notebook or Python script.

Run each cell step by step to:

Preprocess the MNIST dataset

Train different ML/DL models

Evaluate and visualize the results

Example to run the CNN model:

from cnn_model import train_cnn
train_cnn()

📊 Results

Achieved high accuracy with CNN (~99%).

Displayed predictions and confusion matrices for test data.

Analyzed misclassified digits to improve understanding of model behavior.

📂 Project Structure
handwritten-digit-recognition/
│
├── data/                   # Dataset (MNIST or scripts to download)
├── notebooks/              # Jupyter notebooks
├── models/                 # Saved ML/DL models
├── scripts/                # Python scripts for training/evaluation
├── requirements.txt        # Python dependencies
└── README.md

🔗 References

MNIST Dataset

TensorFlow Documentation

Scikit-learn Documentation

⚡ Author
VenushanT



---


Handwritten Recognition with MNIST    :This project focuses on building a machine learning model to automatically recognize handwritten digits (0–9) using the widely-used MNIST dataset. The MNIST dataset contains 70,000 grayscale images of handwritten digits, each of size 28x28 pixels
