# Machine Learning Portfolio

This portfolio showcases the key projects I completed during my Machine Learning internship. Each project demonstrates an end-to-end workflow including data preprocessing, model development, evaluation and deployment using simple applications where applicable.

The projects cover different ML domains including classification, clustering, text analytics and a small web-based interface for model predictions.

---

## 📂 Projects Included

### 1️⃣ Heart Disease Prediction (Classification)
- **Goal:** Predict the presence of heart disease using medical attributes.
- **Tech:** Python, pandas, scikit-learn, matplotlib
- **Key Work:**
  - Exploratory data analysis and preprocessing
  - Logistic Regression and other models
  - Accuracy, confusion matrix, ROC-AUC

📁 Folder: `heart-disease-prediction/`

---

### 2️⃣ Customer Segmentation (Clustering)
- **Goal:** Group customers based on purchasing behavior.
- **Dataset:** Mall Customers data
- **Tech:** K-Means, PCA, matplotlib
- **Key Work:**
  - Data scaling and clustering
  - Dimensionality reduction for visualization
  - Business interpretation of clusters

📁 Folder: `customer-segmentation-clustering/`

---

### 3️⃣ Breast Cancer Diagnosis (Feature Selection + Classification)
- **Goal:** Predict whether tumors are benign or malignant.
- **Dataset:** Breast Cancer Wisconsin
- **Tech:** Feature engineering, Random Forest, Logistic Regression
- **Key Work:**
  - Feature selection techniques
  - Model comparison using precision, recall and F1-score

📁 Folder: `breast-cancer-feature-selection/`

---

### 4️⃣ SMS Spam Detection (NLP Text Classification)
- **Goal:** Classify SMS messages as spam or legitimate (ham).
- **Tech:** TF-IDF, Naive Bayes, scikit-learn
- **Key Work:**
  - Text cleaning and vectorization
  - Building a classifier for real-time use

📁 Folder: `sms-spam-text-classification/`

---

### 5️⃣ Fake News Detection (NLP + Deployment Script)
- **Goal:** Detect misleading or fake news articles based on text.
- **Tech:** Machine Learning, TF-IDF, Streamlit/Flask (simple GUI)
- **Key Work:**
  - End-to-end NLP pipeline
  - Web-based interface for predictions

📁 Folder: `fake-news-detection/`

---

## ▶️ How to Run Any Project

```bash
# Clone the repository
git clone https://github.com/<your-username>/ml-portfolio

cd <project-folder>

# Optional: create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate   # macOS/Linux

# Install required packages
pip install -r requirements.txt

# Run the notebook or app
jupyter notebook
python app.py
