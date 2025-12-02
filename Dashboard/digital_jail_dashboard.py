import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and transformers
model_text = joblib.load(r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\model_text.pkl")
model_num = joblib.load(r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\model_num.pkl")
vectorizer = joblib.load(r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\tfidf_vectorizer.pkl")
scaler = joblib.load(r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\numeric_scaler.pkl")

# Load dataset
df = pd.read_csv(r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\Digital_Jail_Motor_Log_Dataset(final).csv")

st.title("⚙️ DIGITAL JAIL - MOTOR HEALTH MONITORING DASHBOARD")
st.write("AI-based real-time fault detection and quarantine system")

# Fault threshold
threshold = 0.45

# Preprocess features
df["Current_Ratio"] = df["Measured_Current_A"] / df["Rated_Current_A"]

X_text = vectorizer.transform(df["Message"].astype(str))
numeric = df[["Current_Ratio", "Alarm_Duration_s", "Reset_Count"]].values
X_numeric = scaler.transform(numeric)

# Predict probabilities
fault_index_t = list(model_text.classes_).index("fault")
fault_index_n = list(model_num.classes_).index("fault")

P_text = model_text.predict_proba(X_text)[:, fault_index_t]
P_num = model_num.predict_proba(X_numeric)[:, fault_index_n]

df["Fault_Probability"] = (P_text + P_num) / 2
df["Digital_Jail_Decision"] = np.where(df["Fault_Probability"] >= threshold, "QUARANTINE", "ALLOW")

# Display filters
machine_filter = st.selectbox("Select Machine ID", ["ALL"] + sorted(df["Machine_ID"].unique()))

if machine_filter != "ALL":
    df = df[df["Machine_ID"] == machine_filter]

# Color formatting
def color_decision(val):
    color = "red" if val == "QUARANTINE" else "green"
    return f"color: {color}; font-weight: bold;"

st.dataframe(df[["Machine_ID", "StationName", "Fault_Probability", "Digital_Jail_Decision"]]
             .style.applymap(color_decision, subset=["Digital_Jail_Decision"]))
