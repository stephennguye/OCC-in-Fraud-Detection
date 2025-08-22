# Fraudulent Transaction Detection (One-Class Classification)

Detecting anomalies in highly imbalanced credit-card data using **One-Class Classification (OCC)** and deep OCC. Built on the IEEE-CIS Fraud Detection dataset.

---

## 🎯 Motivation
Anomaly detection plays a critical role in identifying rare and irregular patterns within large datasets, particularly in domains such as fraud detection where labeled anomalies are scarce. One-Class Classification (OCC) techniques have emerged as effective solutions for these scenarios, focusing on learning patterns of normal data to detect outliers. In this project, we investigate the application of OCC to credit card fraud detection using the highly imbalanced IEEE-CIS Fraud Detection dataset, where only 3.5% of transactions are fraudulent. We evaluate both classical OCC methods—OneClass SVM, Isolation Forest, and Elliptic Envelope—and deep learning-based models such as Autoencoders and Deep SVDD. 

---

## 🗂️ Data
- Source: IEEE-CIS Fraud Detection (Kaggle) — transaction + identity tables joined by `TransactionID`.
- Scale (after join): ~590k rows, 400+ features; reduced to ~350 after dropping >80%-null columns.
- Challenge: severe class imbalance (~3–4% fraud).

---

## 🔧 Pipeline
1. **Preprocessing**
   - Drop high-null columns; targeted imputations/grouping for domains (`P_emaildomain`, `addr*`, `dist*`).
   - Encode categoricals (One-Hot/Ordinal), **standardize** numerics.
2. **Dimensionality Reduction**
   - **Incremental PCA** (e.g., 10–15 comps depending on model) to stabilize OCC and speed up training.
3. **Train / Validation**
   - **Train only on normal** transactions.
   - Validate on a mixed holdout; report **ROC-AUC** and **Average Precision (AUPRC)**.
4. **OCC ML and DL model training**
   
---

## 🧠 Models
**Classical OCC**
- Isolation Forest
- One-Class SVM (RBF)
- Elliptic Envelope (robust covariance)
- LOF (novelty=True)
- kNN anomaly score

**Deep OCC**
- Autoencoder (feed-forward, symmetric)
- Deep SVDD

Hyperparameters selected via grid search per model.

---

## 📊 Results (validation)
| Model             | ROC-AUC | Avg Precision |
|-------------------|:------:|:-------------:|
| **Isolation Forest** | **0.704** | **0.115** |
| Deep SVDD         | 0.677  | 0.108 |
| Autoencoder       | 0.656  | 0.108 |
| kNN (OCC)         | 0.655  | 0.122 |
| Elliptic Envelope | 0.675  | 0.083 |
| OCSVM (RBF)       | 0.644  | 0.112 |
| LOF               | 0.643  | 0.060 |

**Takeaway:** Unsupervised/OCC fraud is hard; **Isolation Forest** led overall, while **deep OCC** was competitive. Precision-recall remains modest due to extreme overlap and imbalance.

---

## ➕ Extension: GAN Augmentation
- **21-feature** fraud generator (7 curated + **14 PCA** dims) to enrich rare class.
- Synthetic samples moderately realistic (overlap in PCA space), useful for hybrid/semi-supervised follow-ups.

---

## 🛠️ Tech Stack
Python · pandas · scikit-learn · imbalanced-learn · TensorFlow/Keras · matplotlib/plotly

---

## Group 
- Phạm Duy Anh   
- Lê Quốc Anh  
- Nguyễn Thái Nhất Anh  
- Phạm Quang Vinh 
