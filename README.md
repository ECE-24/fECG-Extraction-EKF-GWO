# Extended Kalman Filter with Grey Wolf Optimization for Fetal ECG Extraction

This repository contains the official, reproducible Python code implementation for the journal manuscript:
**"Non-Invasive Fetal ECG Extraction Using Adaptive EKF and GWO Hyperparameter Tuning"**

## 📌 Repository Structure
```text
fECG-Extraction-EKF-GWO/
│
├── data/              <-- Raw and processed PhysioNet records
├── src/               <-- Main Python source scripts (EKF, GWO, Data Loader)
├── results/           <-- Performance output tables and visualization figures
├── notebooks/         <-- Demo Jupyter Notebook for interactive execution
├── main.py            <-- Single-command execution script
├── requirements.txt   <-- Python dependencies
└── README.md          <-- Project documentation
```

## 🚀 Quick Setup & Execution

### 1. Clone or Download Repository
```bash
git clone https://github.com/yourusername/fECG-Extraction-EKF-GWO.git
cd fECG-Extraction-EKF-GWO
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline & Reproduce Results
```bash
python main.py
```

## 📊 Dataset
This project uses the publicly available PhysioNet Non-Invasive Fetal ECG Database (`nifecgdb`). 
Signals are automatically downloaded using the official `wfdb` library during execution.
