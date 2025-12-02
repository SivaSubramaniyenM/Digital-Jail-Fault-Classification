import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.utils import resample
import joblib

file_path = r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\Digital_Jail_Motor_Log_Dataset(final).csv"
df = pd.read_csv(file_path)

# SPLIT FIRST
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["Status"])

# OVERSAMPLE ONLY TRAIN DATA
fault_train = train_df[train_df["Status"] == "fault"]
legit_train = train_df[train_df["Status"] == "legit"]

legit_upsampled = resample(
    legit_train,
    replace=True,
    n_samples=len(fault_train),
    random_state=42
)

train_df_balanced = pd.concat([fault_train, legit_upsampled]).sample(frac=1, random_state=42)

# TEXT FEATURE
vectorizer = TfidfVectorizer(stop_words='english')
X_text_train = vectorizer.fit_transform(train_df_balanced["Message"].astype(str))

# NUMERIC FEATURE
train_df_balanced["Current_Ratio"] = train_df_balanced["Measured_Current_A"] / train_df_balanced["Rated_Current_A"]
numeric_train = train_df_balanced[["Current_Ratio", "Alarm_Duration_s", "Reset_Count"]].values

scaler = StandardScaler()
X_numeric_train = scaler.fit_transform(numeric_train)

y_train = train_df_balanced["Status"]

# TRAIN MODELS
model_text = MultinomialNB().fit(X_text_train, y_train)
model_num = GaussianNB().fit(X_numeric_train, y_train)

# SAVE MODELS & TRANSFORMERS
joblib.dump(model_text, r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\model_text.pkl")
joblib.dump(model_num, r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\model_num.pkl")
joblib.dump(vectorizer, r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\tfidf_vectorizer.pkl")
joblib.dump(scaler, r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\numeric_scaler.pkl")

print("\n✅ Models Saved Successfully!")
