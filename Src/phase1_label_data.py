import pandas as pd

# Load dataset
file_path = r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\Digital_Jail_Motor_Log_Dataset(in).csv"
df = pd.read_csv(file_path)

# Ensure numeric fields are numeric
df["Alarm_Duration_s"] = pd.to_numeric(df["Alarm_Duration_s"], errors="coerce")
df["Reset_Count"] = pd.to_numeric(df["Reset_Count"], errors="coerce")
df["Rated_Current_A"] = pd.to_numeric(df["Rated_Current_A"], errors="coerce")
df["Measured_Current_A"] = pd.to_numeric(df["Measured_Current_A"], errors="coerce")

# Status labeling function
def determine_status(row):
    if row["Alarm_Duration_s"] > 10:
        return "fault"
    if row["Reset_Count"] >= 3:
        return "fault"
    if row["Measured_Current_A"] > (row["Rated_Current_A"] * 1.35):
        return "fault"
    return "legit"

# Apply labeling
df["Status"] = df.apply(determine_status, axis=1)

# Save final dataset
out_path = r"D:\Amrita\ML\CaseStudy2 - Digital_Jail\Digital_Jail_Motor_Log_Dataset(final).csv"
df.to_csv(out_path, index=False)

print("\n✅ Phase 1 Completed.")
print(f"Saved labeled dataset to:\n{out_path}")

# Quick summary
print("\nLabel Distribution:")
print(df["Status"].value_counts())
