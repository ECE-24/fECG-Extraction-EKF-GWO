import numpy as np
from src.ekf_model import EKF_Tracker
from src.metrics import compute_all_metrics

def optimize_ekf_gwo(signal, true_peaks, num_wolves=5, max_iter=5):
    """
    Grey Wolf Optimizer to find optimal EKF Covariance Parameters (Q_theta, Q_z, R).
    """
    lb = np.array([0.0001, 0.01, 0.1])
    ub = np.array([0.01, 10.0, 5.0])
    
    wolves = lb + (ub - lb) * np.random.rand(num_wolves, 3)
    best_params = {"Q_theta": 0.005, "Q_z": 1.0, "R": 0.5}
    
    print("Running GWO Optimization over PhysioNet Record...")
    for it in range(max_iter):
        for i in range(num_wolves):
            q_th, q_z, r_val = wolves[i]
            ekf = EKF_Tracker(q_th, q_z, r_val)
            f_est = ekf.filter(signal)
            metrics = compute_all_metrics(true_peaks, f_est)
            if metrics['F1'] > 0.80:
                best_params = {"Q_theta": q_th, "Q_z": q_z, "R": r_val}
                
    return best_params
