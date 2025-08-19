<p align="center" style="font-family:cursive; color:orange; font-size:120%;">
UCD College Dublin
</p>

<h1 align="center">💸 Fair Loan Default Prediction 💸</h1>

<p align="center" style="color:#159364; font-family:cursive; font-size:100%;">
Data & Computational Science — Final Project
</p>

---

<p align="center">
  <img src="https://www.cashe.co.in/wp-content/uploads/2024/09/Loan-defaults-.png" 
       alt="Loan Default" width="550" height="200">
</p>

---

## 🎯 Business Objective
This project builds a **predictive model** to classify applicants as **defaulters** or **non-defaulters** based on their application data.  
The goal is to **minimize credit risk** for banks while ensuring the decision process remains **fair, unbiased, and transparent**.

The analysis balances **predictive performance** with **responsible AI practices**, including:
- **Fairness auditing** using `fairkit-learn` to detect and mitigate bias.  
- **SHAP explainability** to reveal key drivers of predictions, both globally and per applicant.  

### 🔑 Key Goals
- **Improve risk assessment** with accurate default prediction.  
- **Ensure fairness** across gender, region, and employment type.  
- **Support compliance** with ethical & regulatory standards.  

### 📏 Success Metrics
- **Primary**: Recall (True Positive Rate) for defaulters, Average Precision (PR-AUC).  
- **Secondary**: ROC-AUC, Precision@threshold, F1-score, KS statistic.  
- **Fairness**: Evaluate disparity indices (equal opportunity, demographic parity).  

---

## 📘 Notebook Objectives
This notebook demonstrates how to:
- Clean & preprocess credit application data.  
- Explore data (univariate, bivariate, multivariate analysis).  
- Handle severe class imbalance.  
- Build a **reusable ML pipeline** for multiple algorithms.  
- Optimize hyperparameters with **Optuna**.  
- Explain predictions with **SHAP** & audit **fairness** across groups.

## 🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,scikitlearn,tensorflow" height="55">
  <br>
  <img src="https://skillicons.dev/icons?i=git,github,anaconda" height="55">
</p>

### Core Libraries
- **Data Handling**: `pandas`, `numpy`  
- **Modeling**: `scikit-learn`, `xgboost`, `cuML`, `imblearn`, `thundersvm`  
- **Optimization**: `optuna` , `MLflow`
- **Visualization**: `matplotlib`, `seaborn`, `shap`  
- **Fairness**: `fairkit-learn` 

### ⚡ Models Implemented
1. SVM (RBF kernel)  
2. XGBoost    
3. Logistic Regression  
4. Random Forest  

## 📊 Results Snapshot

| Model            | ROC-AUC | PR-AUC | Recall (Defaulters) | Precision@0.35 | F1-score |
|------------------|:------:|:------:|:-------------------:|:--------------:|:--------:|
| **XGBoost**      | 0.76   | 0.23   | 0.64                 | 0.36           | 0.42     |
| **Random Forest**| 0.73   | 0.19   | 0.58                 | 0.31           | 0.38     |
| **SVM (RBF)**    | 0.74   | 0.21   | 0.60                 | 0.33           | 0.39     |

👉 **Fairness Audit (example):**  
- **Recall** balanced (~92% women vs ~96% men).  
- **Approval Rate** disparity: men receive ~65% as many approvals.  

---

### 🏦 Tier-wise Lending Policy Results

We also tested a **3-tier credit policy** where applicants were ranked by predicted risk and categorized into tiers:  

| Tier       | Policy Action              | Default Rate (%) | Share of Applicants |
|------------|---------------------------|:----------------:|:-------------------:|
| **Tier 1** | Approved (low risk)        | ~5%              | ~55%                |
| **Tier 2** | Review / Conditional loan  | ~18%             | ~30%                |
| **Tier 3** | Rejected (high risk)       | ~42%             | ~15%                |

📌 Tiering helped align **business goals with risk appetite**, reducing high-risk approvals while maintaining coverage of genuine applicants.

➡️ **View notebook outputs here:** [fair_loan_default_prediction.html](notebooks/fair_loan_default_prediction.html)




