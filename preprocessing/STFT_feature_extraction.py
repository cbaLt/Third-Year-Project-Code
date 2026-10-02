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
import train_test_split as helper

def feature_extraction(electrode_num, spectrum, freq_bins, iteration, subject, exercise, mutex):

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

    # global dictionary
    fs = data_dict[database]['fs']

    freq_arr = np.linspace(0, fs/2, freq_bins)
    num_windows = spectrum.shape[0]

    # 7 features per block of spectrogram
    features = np.zeros((num_windows, 7), dtype=np.float32)
    for i in range(num_windows):
        segment = spectrum[i]  # shape: (time, freq)
        print(f'{exercise}, S{subject}, {iteration}; Electrode {electrode_num}: Segment {i+1}/{num_windows}')

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

        # Power Spectrum Ratio (PSR) - frequency resolution is 5 Hz for STFT
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

    # save features from each electrode to hard drive - temporary
    mutex.acquire()
    print(f'Electrode {electrode_num} Acquired Mutex.')
    print(features.shape)
    os.makedirs(os.path.join(f'{database}', 'STFT', 'temp'), exist_ok=True)
    path = os.path.join(f'{database}', 'STFT', 'temp', f'electrode{electrode_num}_stft.npy')
    np.save(path, features)
    print(f'Electrode {electrode_num} Released Mutex.///////////////////////////////////////////////////////////////////////////////////')
    mutex.release()

def create_dataset(iteration, length, num_electrodes, exercise_num, subject_num):

    """
    It is just used to reload the electrode data saved to hard disk and put it into a single file.

    Parameters:
        All parameters are just for formatting.
 
    Returns:
        None: Saves one file to hard disk in subdirectory corresponding to the dataset.
    """

    # empty arrays to store reloaded data
    peak_freq = np.zeros((num_electrodes, length), dtype=np.float32)
    psr = np.zeros((num_electrodes, length), dtype=np.float32)
    mean_power = np.zeros((num_electrodes, length), dtype=np.float32)
    centroid = np.zeros((num_electrodes, length), dtype=np.float32)
    var_freq = np.zeros((num_electrodes, length), dtype=np.float32)
    skew_freq =  np.zeros((num_electrodes, length), dtype=np.float32)
    kurt_freq =  np.zeros((num_electrodes, length), dtype=np.float32)

    for i in range(num_electrodes):
        electrode = i+1

        # reload electrode data and store each feature in corresponding array
        curr_path = os.path.join(f'{database}', 'STFT', 'temp', f'electrode{electrode}_stft.npy')
        combined = np.load(curr_path)

        peak_freq[i] = combined[:, 0]
        mean_power[i] = combined[:, 1]
        centroid[i] = combined[:, 2]
        var_freq[i] = combined[:, 3]
        skew_freq[i] = combined[:, 4]
        kurt_freq[i] = combined[:, 5]
        psr[i] = combined[:, 6]

        # remove temporary electrode file
        os.remove(curr_path)

    final_path = os.path.join(f'{database}', 'STFT', f'{iteration}_{exercise_num}_S{subject_num}_stft.h5')
    # save all features in single file
    with h5py.File(final_path, 'w') as f:
        f.create_dataset('peak freq', data=peak_freq, compression='gzip', compression_opts=8)
        f.create_dataset('mean freq', data=centroid, compression='gzip', compression_opts=8)
        f.create_dataset('psr', data=psr, compression='gzip', compression_opts=8)
        f.create_dataset('mean power', data=mean_power, compression='gzip', compression_opts=8)
        f.create_dataset('skew freq', data=skew_freq, compression='gzip', compression_opts=8)
        f.create_dataset('kurt freq', data=kurt_freq, compression='gzip', compression_opts=8)
        f.create_dataset('var freq', data=var_freq, compression='gzip', compression_opts=8)


def format_data(label, data, trials, num_getures, training_trials, test_trials):

    # used to make main function simpler - simply calls helper functions to format train/test split

    # global dictionary
    M = data_dict[database]['window length']
    step = data_dict[database]['step']

    # split and window data 
    trial_split_index = helper.split_trials(trials, num_gestures, training_trials, test_trials)
    train_labels, test_labels = helper.split_labels(trial_split_index, label, training_trials, test_trials)
    train_time_series, test_time_series = helper.split_time_series(trial_split_index, data, len(train_labels), len(test_labels), training_trials, test_trials)

    return train_time_series, test_time_series

database = 'DB4'
data_dict = {
        'DB1': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 100, 'electrodes': 10, 'subjects': 27, 'train': [1, 3, 4, 6, 7, 8, 9], 'test': [2, 5, 10], 'window length': 20, 'step': 1},
        'DB2': {'E1': 18, 'E2': 24, 'E3': 10, 'fs': 100, 'electrodes': 12, 'subjects': 40, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 400, 'step': 20},
        'DB3': {'E1': 18, 'E2': 24, 'E3': 10, 'fs': 200, 'electrodes': 12, 'subjects': 11, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB4': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 12, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB5': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 16, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2}

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

            emg_path = os.path.join(f'{database}', 'Electrode Data', f'S{subject_num}_{exercise}_A1.mat')
            file = sio.loadmat(emg_path)
            emg_data = file['emg'].T
            labels = file['restimulus'].flatten()
            trials = file['rerepetition'].flatten()

            # downsample from 2 kHz to 200 Hz
            if database in ['DB3', 'DB4']:
                downsample_factor = 10
            
                cutoff_freq = fs/2 - 1
                order = 5
                original_fs = 2000
                # low-pass Butterworth filter
                b, a = signal.butter(order, cutoff_freq / (original_fs / 2), btype='low')
                signals_filtered = np.zeros_like(emg_data)
                for i in range(emg_data.shape[0]):
                    signals_filtered[i, :] = signal.filtfilt(b, a, emg_data[i, :])

                emg_data = signals_filtered[:, ::downsample_factor]
                labels = labels[::downsample_factor]
                trials = trials[::downsample_factor]

            train_data, test_data = format_data(labels, emg_data, trials, num_gestures, train_trials, test_trials)

            train_num_windows = int(((train_data.shape[1] - W) // step) + 1)
            test_num_windows = int(((test_data.shape[1] - W) // step) + 1)
            num_freq_bins = int((W / 2) + 1)
            train_freq = np.zeros((num_electrodes, train_num_windows, num_freq_bins), dtype=np.float32)
            test_freq = np.zeros((num_electrodes, test_num_windows, num_freq_bins), dtype=np.float32)

            for num in range(num_electrodes):
                f, t, train_result = signal.stft(train_data[num], window='hann', fs=fs, nperseg=W, noverlap=W-step, padded=False, boundary=None)
                train_freq[num] = (np.abs(train_result)**2).T

                f, t, test_result = signal.stft(test_data[num], window='hann', fs=fs, nperseg=W, noverlap=W-step, padded=False, boundary=None)
                test_freq[num] = (np.abs(test_result)**2).T

            os.makedirs(os.path.join(f'{database}', 'STFT'), exist_ok=True)
            print(train_freq.shape, test_freq.shape)
            
            # multithreading - create 1 thread for each electrode
            process = []
            manager = mp.Manager()
            mutex = manager.Lock()
 
            for electrode, data in enumerate(train_freq):

                thread = mp.Process(target=feature_extraction, args=(electrode+1, data, num_freq_bins, 'Train', subject_num, exercise, mutex))
                process.append(thread)

                thread.start()

            for p in process:
                p.join()
        
            # reload electrode data and store in single file
            create_dataset('Training', train_freq.shape[1], num_electrodes, exercise, subject_num)

            process_two = []
            mutex2 = manager.Lock()

            # to create test features
            for electrode, data in enumerate(test_freq):

                thread = mp.Process(target=feature_extraction, args=(electrode+1, data, num_freq_bins, 'Test', subject_num, exercise, mutex2))
                process_two.append(thread)

                thread.start()

            for p in process_two:
                p.join()

            # reload electrode data to store in single file
            create_dataset('Test', test_freq.shape[1], num_electrodes, exercise, subject_num)
