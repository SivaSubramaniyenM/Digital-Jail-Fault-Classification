# Dataset Information

The dataset used in this project contains **PLC alarm logs** and **motor operating parameters** recorded from an industrial production line.  
It includes both text-based events and numeric sensor values that are essential for fault classification.

The dataset **is not included** in this repository due to confidentiality and size constraints.

---

## Expected Dataset Format

Your CSV file should contain the following columns:

### Required Log Columns
- `Timestamp`
- `Machine_ID`
- `Message`
- `Alarm_Duration_s`
- `Reset_Count`

### Motor Parameter Columns
- `Measured_Current_A`
- `Rated_Current_A`
- `Measured_Voltage_V`
- `Measured_PowerFactor`

### Derived Columns (computed automatically)
- `Current_Ratio = Measured_Current_A / Rated_Current_A`

---

## How to Use Your Own Dataset

1. Prepare your dataset following the structure above.  
2. Save it as: Digital_Jail_Motor_Log_Dataset(final).csv
3. Place it inside this folder: Digital-Jail-Fault-Classification/Data/
4. The dashboard and training scripts will automatically load it.

---

## Privacy Note

Real industrial data often contains:
- Time stamps
- Machine identifiers
- Fault patterns  
These may reveal operational details.  
Therefore, the dataset must **not** be uploaded to GitHub repositories.

---

## Important
If you retrain the model:
- Delete the existing `.pkl` files in `/Models/`
- Re-run the scripts inside `/Src/`

This will regenerate updated model files.




