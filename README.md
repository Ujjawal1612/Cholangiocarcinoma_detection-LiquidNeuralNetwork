# Breast-cancer-detection-using-LNN
# Cholangiocarcinoma Detection using Liquid Neural Network with Custom Parrot Optimizer

## 📌 Overview
This project focuses on detecting **cholangiocarcinoma (bile duct cancer)** from histopathological images using a **Liquid Neural Network (LNN)** optimized with a **custom Parrot Optimization Algorithm (POA)**.  
The model is designed to assist medical professionals by providing an automated, accurate, and efficient cancer diagnosis system.

---

## 🚀 Features
- **Liquid Neural Network (LNN)** for dynamic and adaptive learning.
- **Custom Parrot Optimizer (POA)** for optimal parameter tuning.
- **Data augmentation** to improve generalization.
- Handles **class imbalance** for better accuracy in medical datasets.
- **Comprehensive evaluation metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC AUC Curve
- **Image-level predictions** for real-world usability.

---

## 📂 Dataset
- **Type:** Histopathological images of cholangiocarcinoma and non-cancerous tissue.
- **Source:** *(Add dataset link here — e.g., Kaggle, TCGA, or any other)*  
- **Preprocessing Steps**:
  - Image resizing and normalization
  - Augmentation (rotation, flipping, zoom)
  - Class balancing techniques

---

## 🛠️ Technologies Used
- **Language:** Python  
- **Libraries & Frameworks**:
  - PyTorch
  - NumPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn
- **Optimization:** Custom Parrot Optimization Algorithm (POA)

---

## 🔄 Workflow
1. **Data Preprocessing** – Cleaning, resizing, augmentation.  
2. **Model Building** – Implementing Liquid Neural Network in PyTorch.  
3. **Optimization** – Applying POA for hyperparameter tuning.  
4. **Training & Validation** – Tracking metrics to prevent overfitting.  
5. **Evaluation** – Generating accuracy, confusion matrix, and ROC curve.  
6. **Prediction** – Testing on unseen histopathology images.

---

## 📊 Results
| Metric        | Value |
|--------------|-------|
| Accuracy     | XX%   |
| Precision    | XX%   |
| Recall       | XX%   |
| F1-score     | XX%   |
| ROC AUC      | XX%   |

*(Replace `XX%` with your actual results after training.)*

---

## 📷 Sample Predictions
*(Insert prediction images here with annotations.)*

---

## 📦 Installation & Usage
```bash
# Clone repository
git clone https://github.com/<your-username>/Cholangiocarcinoma-LNN-POA.git

# Navigate to folder
cd Cholangiocarcinoma-LNN-POA

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Predict on a new image
python predict.py --image path_to_image.jpg
