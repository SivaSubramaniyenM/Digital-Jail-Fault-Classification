# Digital-Jail-Fault-Classification

A Machine Learning–powered industrial fault classification system implementing the **Digital Jail** concept for automated maintenance lockout.  
This project uses a **Hybrid Naïve Bayes** approach to classify PLC alarm logs and motor operating parameters into *legit* or *fault* states, preventing repeated unsafe resets and enforcing preventive maintenance in automated manufacturing environments.

---

## Project Overview

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

## Machine Learning Architecture

### Hybrid Naïve Bayes Classifier
- **MultinomialNB** → Alarm message text classification  
- **GaussianNB** → Numeric features (Current Ratio, Alarm Duration, Reset Count)  
- **Final Fault Score** = Average of both probabilities  
- **Decision Logic**:  Fault_Probability ≥ 0.45 → QUARANTINE | Else → ALLOW

---

## Streamlit Dashboard

A real-time dashboard to monitor:

- Machine ID  
- PLC alarm message  
- Calculated fault probability  
- Digital Jail decision (QUARANTINE or ALLOW)  
- Filtering by machines  

Run locally: streamlit run Dashboard/digital_jail_dashboard.py

---

## Installation & Setup

### 1. Clone the repository: git clone https://github.com/SivaSubramaniyenM/Digital-Jail-Fault-Classification.git 
cd Digital-Jail-Fault-Classification

--- 

### 2. Install dependencies:
pip install -r requirements.txt

---

### 3. Add dataset (instructions inside `/Data/README.md`)

### 4. Run dashboard: streamlit run Dashboard/digital_jail_dashboard.py

---

## Dataset Notice

A custom industrial-style dataset was developed by modelling PLC alarm messages and motor electrical characteristics, reflecting the complexity of real production environments. It is not included in this repository due to its size.  
To run the project, place your dataset in the /data folder following the same column structure.

Instructions and format details are provided in `/Data/README.md`.

---

## License

This project is released under the **MIT License**. Feel free to use, modify, and adapt it.

---

## Author

**Siva Subramaniyen M** 
Electrical Engineer,
M.Tech Artificial Intelligence, Amrita Vishwa Vidyapeetham



