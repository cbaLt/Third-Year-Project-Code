import numpy as np
import scipy.io as sio
import pandas as pd
import multiprocessing as mp
import os
import scipy.sparse as sp
import scipy.signal as signal
import h5py
import gc
import time

def feature_extraction(electrode_num, spectrum, freq_bins, subject, exercise, mutex):

    """
    Extract 7 features from HHT power spectrogram for each electrode.

    Parameters:
        Electrode number (int): Used to print to terminal to keep track.
        spectrum (2D array): The power spectrum - 2D array with shape (t, freq)
        Frequency bins (int): Number of frequency bins - bad naming, sorry.
        Iteration (string): Either Training or Test - to display to terminal.
        Exercise, Subject (strings): To display to terminal what exercise and subject is being processed.
        Mutex: Used to prevent multiple threads saving to disk at the same time.
 
    Returns:
        None: It saves the 7 features for each window for each electrode to main memory as numpy file.
    """

    fs = data_dict[database]['fs']

    freq_arr = np.linspace(0, fs/2, freq_bins)
    num_windows = spectrum.shape[0]

    # 7 features per block of spectrogram
    features = np.zeros((num_windows, 7), dtype=np.float32)
    for i in range(num_windows):
        segment = spectrum[i]  # shape: (time, freq)
        print(f'{exercise}, S{subject}; Electrode {electrode_num}: Segment {i+1}/{num_windows}')

        # Peak frequency
        peak_idx = np.argmax(segment)
        peak_freq = freq_arr[peak_idx]

        # Mean power
        mean_power = np.mean(segment)

        # Spectral moments
        segment_sum = np.sum(segment)
        freq_slice = freq_arr

        if segment_sum > 0:
            centroid = np.sum(segment * freq_slice) / segment_sum
            var_freq = np.sum(segment * (freq_slice - centroid) ** 2) / segment_sum
            skew_freq = np.sum(segment * (freq_slice - centroid) ** 3) / (segment_sum * var_freq ** 1.5 + 1e-12)
            kurt_freq = np.sum(segment * (freq_slice - centroid) ** 4) / (segment_sum * var_freq ** 2 + 1e-12)
        else:
            centroid, var_freq, skew_freq, kurt_freq = 0, 0, 0, 0

        # Power Spectrum Ratio (PSR) - 5 Hz is resolution
        freq_resolution = freq_arr[1] - freq_arr[0]  # Assuming uniform spacing
        f_interest = int(5 / freq_resolution)

        local_start = max(peak_idx - f_interest, 0)
        local_end = min(peak_idx + f_interest + 1, len(segment))
        psr = np.sum(segment[local_start:local_end]) / (segment_sum + 1e-12)

        # Assign features
        features[i] = np.array([
            peak_freq, mean_power,
            centroid, var_freq, skew_freq, kurt_freq,
            psr
        ], dtype=np.float32)

    # critical section when saving features from each electrode to hard disk
    mutex.acquire()
    print(f'Electrode {electrode_num} Acquired Mutex.')
    os.makedirs(os.path.join(f'{database}', 'STFT', 'temp'), exist_ok=True)
    path = os.path.join(f'{database}', 'STFT', 'temp', f'electrode{electrode_num}_stft.npy')
    np.save(path, features)
    print(f'Electrode {electrode_num} Released Mutex.///////////////////////////////////////////////////////////////////////////////////')
    mutex.release()

def create_dataset(length, num_electrodes, exercise_num, subject_num):

    # reload temporary electrode features to save in a single file
    
    peak_freq = np.zeros((num_electrodes, length), dtype=np.float32)
    psr = np.zeros((num_electrodes, length), dtype=np.float32)
    mean_power = np.zeros((num_electrodes, length), dtype=np.float32)
    centroid = np.zeros((num_electrodes, length), dtype=np.float32)
    var_freq = np.zeros((num_electrodes, length), dtype=np.float32)
    skew_freq =  np.zeros((num_electrodes, length), dtype=np.float32)
    kurt_freq =  np.zeros((num_electrodes, length), dtype=np.float32)

    for i in range(num_electrodes):
        electrode = i+1

        # reload and save in corresponding feature array
        curr_path = os.path.join(f'{database}', 'STFT', 'temp', f'electrode{electrode}_stft.npy')
        combined = np.load(curr_path)

        peak_freq[i] = combined[:, 0]
        mean_power[i] = combined[:, 1]
        centroid[i] = combined[:, 2]
        var_freq[i] = combined[:, 3]
        skew_freq[i] = combined[:, 4]
        kurt_freq[i] = combined[:, 5]
        psr[i] = combined[:, 6]

        os.remove(curr_path)

    # create file
    final_path = os.path.join(f'{database}', 'STFT', f'{exercise_num}_S{subject_num}_stft.h5')
    with h5py.File(final_path, 'w') as f:
        f.create_dataset('peak freq', data=peak_freq, compression='gzip', compression_opts=8)
        f.create_dataset('mean freq', data=centroid, compression='gzip', compression_opts=8)
        f.create_dataset('psr', data=psr, compression='gzip', compression_opts=8)
        f.create_dataset('mean power', data=mean_power, compression='gzip', compression_opts=8)
        f.create_dataset('skew freq', data=skew_freq, compression='gzip', compression_opts=8)
        f.create_dataset('kurt freq', data=kurt_freq, compression='gzip', compression_opts=8)
        f.create_dataset('var freq', data=var_freq, compression='gzip', compression_opts=8)


# ONLY FOR DB1 
database = 'DB1'
data_dict = {
        'DB1': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 100, 'electrodes': 10, 'subjects': 27, 'train': [1, 3, 4, 6, 7, 8, 9], 'test': [2, 5, 10], 'window length': 20, 'step': 1},
        }

num_electrodes = data_dict[database]['electrodes']
num_subjects = data_dict[database]['subjects']
W = data_dict[database]['window length']
step = data_dict[database]['step']
fs = data_dict[database]['fs']
 
train_trials = data_dict[database]['train']
test_trials = data_dict[database]['test']

if __name__ == '__main__':

    print("Number of cores: ", mp.cpu_count())


    for exercise in ['E1', 'E2', 'E3']:

        num_gestures = data_dict[database][exercise]
 
        for subject_num in range(1, num_subjects+1):

            # load EMG data, trials and labels from Ninapro file
            emg_path = os.path.join(f'{database}', 'Electrode Data', f'S{subject_num}_A1_{exercise}.mat')
            file = sio.loadmat(emg_path)
            emg_data = file['emg'].T
            labels = file['restimulus'].flatten()
            trials = file['rerepetition'].flatten()

            # calculate number of windows from window size 20 and step 1
            num_windows = int(((emg_data.shape[1] - W) // step) + 1)
            num_freq_bins = int((W / 2) + 1)
            frequency = np.zeros((num_electrodes, num_windows, num_freq_bins), dtype=np.float32)

            # compute STFT for each window and store in frequency array
            for num in range(num_electrodes):
                f, t, Zxx = signal.stft(emg_data[num], window='hann', fs=fs, nperseg=W, noverlap=W-step, padded=False, boundary=None)
                frequency[num] = (np.abs(Zxx)**2).T

            # create subdirectory if it doesn't exist
            os.makedirs(os.path.join(f'{database}', 'STFT'), exist_ok=True)
            
            # multithreading - create 1 thread for each electrode
            process = []
            manager = mp.Manager()
            mutex = manager.Lock()
 
            for electrode, data in enumerate(frequency):

                # each electrode calls feature extraction function
                thread = mp.Process(target=feature_extraction, args=(electrode+1, data, num_freq_bins, subject_num, exercise, mutex))
                process.append(thread)

                thread.start()

            for p in process:
                p.join()
        
            # reload temporary electrode files to store in single feature file
            create_dataset(frequency.shape[1], num_electrodes, exercise, subject_num)
