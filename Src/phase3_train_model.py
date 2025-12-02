import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import numpy as np

# Load labeled (but NOT oversampled) dataset
file_path = r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\Digital_Jail_Motor_Log_Dataset(final).csv"
df = pd.read_csv(file_path)

# Train-test split FIRST
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["Status"])

# Oversample ONLY TRAIN data
fault_train = train_df[train_df["Status"] == "fault"]
legit_train = train_df[train_df["Status"] == "legit"]

legit_upsampled = resample(
    legit_train,
    replace=True,
    n_samples=len(fault_train),
    random_state=4
2)

train_df_balanced = pd.concat([fault_train, legit_upsampled]).sample(frac=1, random_state=42)

# TEXT FEATURE
vectorizer = TfidfVectorizer(stop_words='english')
X_text_train = vectorizer.fit_transform(train_df_balanced["Message"].astype(str))
X_text_test = vectorizer.transform(test_df["Message"].astype(str))

# NUMERIC FEATURES
train_df_balanced["Current_Ratio"] = train_df_balanced["Measured_Current_A"] / train_df_balanced["Rated_Current_A"]
test_df["Current_Ratio"] = test_df["Measured_Current_A"] / test_df["Rated_Current_A"]

numeric_train = train_df_balanced[["Current_Ratio", "Alarm_Duration_s", "Reset_Count"]].values
numeric_test = test_df[["Current_Ratio", "Alarm_Duration_s", "Reset_Count"]].values

scaler = StandardScaler()
X_numeric_train = scaler.fit_transform(numeric_train)
X_numeric_test = scaler.transform(numeric_test)

y_train = train_df_balanced["Status"]
y_test = test_df["Status"]

# Train NB Models
model_text = MultinomialNB().fit(X_text_train, y_train)
model_num = GaussianNB().fit(X_numeric_train, y_train)

# Probability of "fault"
fault_class_index_text = list(model_text.classes_).index("fault")
fault_class_index_num = list(model_num.classes_).index("fault")

P_text = model_text.predict_proba(X_text_test)[:, fault_class_index_text]
P_num = model_num.predict_proba(X_numeric_test)[:, fault_class_index_num]

P_final = (P_text + P_num) / 2
y_pred = np.where(P_final > 0.5, "fault", "legit")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Digital Jail Decision (using probability threshold)
threshold = 0.45  # can adjust after observing failure cost
final_output = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Fault_Probability": P_final
})
final_output["Digital_Jail_Decision"] = np.where(final_output["Fault_Probability"] >= threshold, "QUARANTINE", "ALLOW")

print("\nSample Digital Jail Actions:")
print(final_output.head(20))
