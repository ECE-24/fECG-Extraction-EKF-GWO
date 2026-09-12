import numpy as np

class EKF_Tracker:
    """
    Extended Kalman Filter for Maternal ECG Cancellation and Fetal ECG Estimation.
    """
    def __init__(self, Q_theta=0.005, Q_z=1.0, R=0.5):
        self.Q_theta = Q_theta
        self.Q_z = Q_z
        self.R = R

    def filter(self, signal):
        N = len(signal)
        x_hat = np.zeros(N)
        P = 0.1
        for k in range(1, N):
            x_pred = x_hat[k-1]
            P_pred = P + self.Q_theta
            K = P_pred / (P_pred + self.R + 1e-5)
            x_hat[k] = x_pred + K * (signal[k] - x_pred)
            P = (1.0 - K) * P_pred
        return signal - x_hat  # Estimated fECG residual
