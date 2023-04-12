'''
generator = data_generator_subject15()

# Loop through the generator and feed the chunks to your deep learning model
for chunk in generator:
    print(chunk[0].shape)  # The shape of each chunk is (chunk_size, 22)
    # Train your deep learning model using the chunk
    break
    
'''

import os
import numpy as np
import h5py
import shutil
import gc
import time
from scipy.interpolate import interp1d

def data_generator(subject_id, chunk_size=512, overlap=0.5):
    file_path = os.path.join(f'/content/drive/MyDrive/EEG_detection/EEG_data/subject_{subject_id:02d}.mat')
    with h5py.File(file_path, 'r') as f:
        eeg_data_struct = f['eeg_data']
        labels_struct = f['labels']

        # Access the 'filteredData' cell within the 'eeg_data' struct
        dataset = eeg_data_struct['filteredData']
        labels = np.array(labels_struct['sumLabel'])

        total_rows = dataset.shape[0]
        step_size = int(chunk_size * (1 - overlap))

        for start_idx in range(0, total_rows - chunk_size + 1, step_size):
            end_idx = start_idx + chunk_size
            chunk_data = dataset[start_idx:end_idx, :]
            chunk_label = labels[start_idx:end_idx, :]
            yield np.array(chunk_data), np.array(chunk_label)

def data_generator_subject15(chunk_size=512, overlap=0.5):
    file_path = (f'/content/drive/MyDrive/EEG_detection/闹心玩意/f_15.mat')
    with h5py.File(file_path, 'r') as f:
        dataset = f['filteredData']
        total_rows = dataset.shape[0]
        step_size = int(chunk_size * (1 - overlap))

        labels = f['sumLabel']
        # labels = np.array(labels)

        for start_idx in range(0, total_rows - chunk_size + 1, step_size):
            end_idx = start_idx + chunk_size
            chunk_data = dataset[start_idx:end_idx, :]
            chunk_label = labels[start_idx:end_idx, :]
            yield np.array(chunk_data), np.array(chunk_label)

