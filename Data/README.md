# Dataset Information

This project uses a PLC alarm log + motor parameter dataset for fault
classification. The dataset contains:

- Alarm messages and timestamps
- Motor electrical measurements (current, voltage, power factor)
- Derived fault features (duration, reset counts, current ratio)
- Labels: "fault" and "legit"

The dataset is **not included in this repository** due to size and
confidentiality.

## To run the dashboard:
1. Download the dataset file named:
   `Digital_Jail_Motor_Log_Dataset(final).csv`
2. Place this file inside the `data/` folder.
3. Run the Streamlit dashboard
