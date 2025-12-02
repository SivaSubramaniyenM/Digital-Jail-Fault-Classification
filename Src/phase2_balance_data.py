import pandas as pd
from sklearn.utils import resample

file_path = r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\Digital_Jail_Motor_Log_Dataset(final).csv"
df = pd.read_csv(file_path)

# Split
fault_df = df[df["Status"] == "fault"]
legit_df = df[df["Status"] == "legit"]

# Oversample legit → match fault count
legit_upsampled = resample(
    legit_df,
    replace=True,
    n_samples=len(fault_df),
    random_state=42
)

# Combine
balanced_df = pd.concat([fault_df, legit_upsampled])

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
out_path = r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\Digital_Jail_Motor_Log_Dataset(balanced).csv"
balanced_df.to_csv(out_path, index=False)

print("\n✅ Phase 2 Completed — Class Balance Applied")
print(f"Saved balanced dataset to:\n{out_path}")
print("\nNew Label Distribution:")
print(balanced_df['Status'].value_counts())
