# Digital-Jail-Fault-Classification

A Machine Learning–powered industrial fault classification system implementing the **Digital Jail** concept for automated maintenance lockout.  
This project uses a **Hybrid Naïve Bayes** approach to classify PLC alarm logs and motor operating parameters into *legit* or *fault* states, preventing repeated unsafe resets and enforcing preventive maintenance in automated manufacturing environments.

---

## 🚀 Project Overview

Modern industrial machines frequently generate alarms during operation. Some alarms are harmless, while others indicate abnormal motor load, mechanical binding, or electrical faults. Operators often clear alarms repeatedly to keep production running, which can escalate hidden faults into breakdowns.

This system introduces **Digital Jail** — a mechanism that automatically *quarantines* a machine when abnormal patterns are detected, forcing proper inspection before continuation.

The project includes:

- Dataset preparation using PLC logs + motor electrical parameters  
- Automatic fault labeling using domain rules  
- Hybrid Naïve Bayes classification (Text + Numeric)  
- Model training, evaluation, and serialization  
- A **Streamlit-based dashboard** for real-time Digital Jail decisions  
- A clean, modular ML training pipeline under `src/`

---

## 🧠 Machine Learning Architecture

### ✔ Hybrid Naïve Bayes Classifier
- **MultinomialNB** → Alarm message text classification  
- **GaussianNB** → Numeric features (Current Ratio, Alarm Duration, Reset Count)  
- **Final Fault Score** = Average of both probabilities  
- **Decision Logic**:  
