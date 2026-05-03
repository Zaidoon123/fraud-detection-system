# 💳 Credit Card Fraud Detection System

A machine learning system that detects fraudulent credit card transactions using Random Forest, Logistic Regression, and XGBoost — deployed as a REST API with Flask.

---

## 🎯 Problem Statement

Credit card fraud is a major challenge for financial institutions. With highly imbalanced data (only 0.17% of transactions are fraudulent), traditional accuracy metrics are misleading. This project builds a robust fraud detection pipeline that handles class imbalance and optimizes for catching fraud while minimizing false alarms.

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by mlg-ulb
- **Size:** 284,807 transactions
- **Features:** 30 (Time, V1-V28 PCA-transformed, Amount)
- **Target:** Class (0 = Normal, 1 = Fraud)

### Class Distribution:
| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 284,315 | 99.83% |
| Fraud (1) | 492 | 0.17% |


> ⚠️ Dataset not included due to size (143MB)
> Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
> Place `creditcard.csv` in the root folder before running
---

## 🔍 Key Findings from EDA

- No missing values in the dataset
- Fraud transactions have slightly higher average amount ($122) vs normal ($88)
- Fraudsters avoid very large transactions (max fraud = $2,125 vs max normal = $25,691)
- Extreme class imbalance requires special handling before training

---

## ⚙️ Pipeline

```
Raw Data (284,807 transactions)
        ↓
EDA & Analysis
        ↓
Scale Amount & Time (StandardScaler)
        ↓
Train/Test Split (80/20)
        ↓
Handle Imbalance (SMOTE on training data only)
        ↓
Train 3 Models
        ↓
Evaluate with Precision, Recall, F1
        ↓
Deploy Best Model as Flask API
```

---

## 🤖 Models & Results

| Model | Fraud Recall | Fraud Precision | Fraud F1 |
|---|---|---|---|
| Logistic Regression | 0.93 | 0.06 | 0.11 |
| XGBoost | 0.85 | 0.72 | 0.78 |
| **Random Forest** ✅ | **0.84** | **0.91** | **0.87** |

### Why Random Forest won:
- Best balance between Precision and Recall
- 91% Precision → minimal false alarms
- 84% Recall → catches most fraud
- Most suitable for production deployment

### Why not Accuracy?
With 99.83% normal transactions, a model predicting everything as normal achieves 99.83% accuracy but catches 0% of fraud. We use **F1-score and Recall** instead.

---

## 🛠️ Tech Stack

| Tool | Use |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-learn | ML models & preprocessing |
| XGBoost | Gradient boosting model |
| imbalanced-learn | SMOTE for class balancing |
| Flask | REST API |
| joblib | Model serialization |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fraud-detection
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask API
```bash
python app.py
```

### 4. Test the API
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

### Example Response:
```json
{
  "fraud_probability": 0.87,
  "prediction": "FRAUD"
}
```

---

## 📁 Project Structure
```
fraud-detection-system/
├── model/
│   └── fraud_model.pkl
├── app.py
├── Fraud_Detection_System_Project_.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Interactive web dashboard for real-time predictions
- Deploy to cloud (AWS/GCP/Heroku)
- Add more features through feature engineering
- Test with other models (Neural Network, LightGBM)

---

## 👨‍💻 Author

**Zaidoon Hijazeen**
B.Sc. Data Science & Artificial Intelligence — Hashemite University
