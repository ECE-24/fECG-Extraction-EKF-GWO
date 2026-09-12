import wfdb
import numpy as np

def load_physionet_record(record_name='ecgca591', database='nifecgdb'):
    """
    Downloads and loads non-invasive fetal ECG record from PhysioNet.
    """
    try:
        record = wfdb.rdrecord(record_name, pb_dir=database)
        annotation = wfdb.rdann(record_name, 'qrs', pb_dir=database)
        signal = record.p_signal[:, 0]  # First lead abdominal ECG
        fs = record.fs
        fetal_qrs_indices = annotation.sample
        return signal, fs, fetal_qrs_indices
    except Exception as e:
        print(f"Error loading PhysioNet record {record_name}: {e}")
        return None, None, None
