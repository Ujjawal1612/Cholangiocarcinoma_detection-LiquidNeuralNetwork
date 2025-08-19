# LNN-Cholangiocarcinoma-Detection

##  Overview
This project detects **Cholangiocarcinoma (bile duct cancer)** from histopathological images using a **Liquid Neural Network (LNN)** optimized with a **Custom Parrot Optimization Algorithm (POA)**.  
The approach combines the adaptability of LNN with the global search capabilities of POA to achieve high accuracy, robust generalization, and reduced overfitting.

---

##  Features
- **Liquid Neural Network (LNN)** for dynamic, adaptive learning.
- **Custom Parrot Optimization Algorithm (POA)** for hyperparameter tuning.
- **Medical-grade image preprocessing** with cleaning, resizing, and augmentation.
- **Class imbalance handling** to improve real-world accuracy.
- **Comprehensive evaluation metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC AUC Curve
- **Image-level predictions** for clinical usability.

---

##  Dataset
- **Type:** Histopathological images of cholangiocarcinoma and healthy tissue.
- **Source:** *(Add dataset link here — e.g., Kaggle, TCGA, etc.)*
- **Preprocessing Steps**:
  - Image cleaning and resizing
  - Normalization
  - Augmentation (rotation, flipping, zoom)
  - Balancing classes

---

##  Technologies Used
- **Language:** Python  
- **Libraries & Frameworks**:
  - PyTorch
  - NumPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn
- **Optimization:** Custom Parrot Optimization Algorithm (POA)

---

##  Workflow
![Workflow Diagram](workflow_diagram.png)

1. **Data Preprocessing** – Cleaning, resizing, augmentation.  
2. **Model Building** – Implementing Liquid Neural Network in PyTorch.  
3. **Optimization** – Applying **Parrot Optimization Algorithm** for hyperparameter tuning.  
4. **Training & Validation** – Tracking accuracy, loss, and validation metrics to prevent overfitting.  
5. **Evaluation** – Generating accuracy score, confusion matrix, and ROC curve.  
6. **Prediction** – Testing on unseen histopathological images.

---

##  Results
| Metric        | Value |
|--------------|-------|
| Accuracy     | XX%   |
| Precision    | XX%   |
| Recall       | XX%   |
| F1-score     | XX%   |
| ROC AUC      | XX%   |

*(Replace XX% with your actual model performance.)*

---

##  Sample Predictions
*(Insert example prediction images here.)*

---

##  Installation & Usage
```bash
# Clone repository
git clone https://github.com/<your-username>/LNN-Cholangiocarcinoma-Detection.git

# Navigate to project folder
cd LNN-Cholangiocarcinoma-Detection

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Predict on a new image
python predict.py --image path_to_image.jpg
