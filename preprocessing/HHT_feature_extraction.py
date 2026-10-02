import numpy as np
import scipy.io as sio
import pandas as pd
import multiprocessing as mp
import os
import scipy.sparse as sp
import h5py
import gc
import time
# custom functions to split train/test data for DB3, DB4 and DB5
import train_test_split as helper

def feature_extraction(electrode_num, spectrum, freq_bins, iteration, subject, exercise, mutex):

    """
    Extract 8 features from HHT power spectrogram for each electrode.

    Parameters:
        Electrode number (int): Used to print to terminal to keep track.
        spectrum (2D array): The power spectrum (not segmented, it is the length of the EMG signal) of the electrode with shape (t, freq bins), since HHT calculates frequency for each time step. 
        Frequency bins (int): Number of frequency bins - bad naming, sorry.
        Iteration (string): Either Training or Test - to display to terminal.
        Exercise, Subject (strings): To display to terminal what exercise and subject is being processed.
        Mutex: Used to prevent multiple threads saving to disk at the same time.
 
    Returns:
        None: It saves the 8 features for each window for each electrode to main memory as numpy file.
    """

    # global information dictionary
    W = data_dict[database]['window length']
    step = data_dict[database]['step']
    fs = data_dict[database]['fs']

    # segment the spectrum - size depends on the dataset sample rate
    windowed_data = helper.rolling_window_HHT_spectrum(spectrum, W, step)

    # create frequency vector from number of frequency bins
    freq_arr = np.linspace(0, fs/2, freq_bins)
    num_windows = windowed_data.shape[0]

    # at the moment it is set to the whole spectrogram, but this can be adjusted to segment the spectrogram and take more features from each 'block'
    block_time = W
    block_freq = freq_bins
    step_time = 1
    step_freq = 1

    time_frames, freq_bins = windowed_data.shape[1], windowed_data.shape[2]

    # Number of blocks
    num_blocks_time = (time_frames - block_time) // step_time + 1
    num_blocks_freq = (freq_bins - block_freq) // step_freq + 1

    # 8 features per block of spectrogram
    features = np.zeros((num_windows, (num_blocks_time*num_blocks_freq), 8), dtype=np.float32)

    for i in range(num_windows):

        segment = windowed_data[i]
        print(f'{exercise}, S{subject}, {iteration}; Electrode {electrode_num}: Segment {i+1}/{num_windows}')

        block_features = []  # store feature vectors per block

        for t in range(0, time_frames - block_time + 1, step_time):
            for f in range(0, freq_bins - block_freq + 1, step_freq):
                block = segment[t:t+block_time, f:f+block_freq]

                # IMF count
                count_imfs = np.count_nonzero(block)
                num_imfs = count_imfs / block.shape[0]

                # max. frequency (peak freq bin)
                peak_freq_bin = np.argmax(block)
                peak_freq_idx = np.unravel_index(peak_freq_bin, block.shape)
                peak_freq = freq_arr[f + peak_freq_idx[1]]  # offset by block starting freq

                # mean power
                mean_power = np.sum(block) / block_freq

                # spectral moments
                block_sum = np.sum(block)
                freq_slice = freq_arr[f:f+block_freq]

                # prevent divide by zero error
                if block_sum > 0:
                    # mean and variance are not standardised
                    centroid = np.sum(np.sum(block, axis=0) * freq_slice) / block_sum
                    var_freq = np.sum(np.sum(block, axis=0) * (freq_slice - centroid)**2) / block_sum

                    # skewness and kurtosis are standardised to variance
                    skew_freq = np.sum(np.sum(block, axis=0) * (freq_slice - centroid)**3) / (block_sum * var_freq**1.5 + 1e-12)
                    kurt_freq = np.sum(np.sum(block, axis=0) * (freq_slice - centroid)**4) / (block_sum * var_freq**2 + 1e-12)
                else:
                    centroid, var_freq, skew_freq, kurt_freq = 0, 0, 0, 0

                # Power Spectrum Ratio (PSR) - frequency resolution is 0.5 Hz for HHT
                freq_resolution = 0.5
                # index corresponding to frequency of 5 Hz
                f_interest = int(5 / freq_resolution)

                # calculate PSR with 5 Hz frequencies around the peak 
                local_start = max(peak_freq_idx[1] - f_interest, 0)
                local_end = min(peak_freq_idx[1] + f_interest + 1, block.shape[1])
                psr = np.sum(block[:, local_start:local_end]) / (np.sum(block) + 1e-12)

                block_features.append([
                    num_imfs, peak_freq, mean_power,
                    centroid, var_freq, skew_freq, kurt_freq,
                    psr
                ])

        features[i] = np.array(block_features, dtype=np.float32)

    # create temporary folder and save it there
    mutex.acquire()
    print(f'Electrode {electrode_num} Acquired Mutex.')
    os.makedirs(os.path.join(f'{database}', 'HHT', 'temp'), exist_ok=True)
    path = os.path.join(f'{database}', 'HHT', 'temp', f'electrode{electrode_num}_hht.npy')
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

    # this number is changed depending on how many blocks were used in feature extraction function
    features_per_block = 1

    # create empty arrays
    peak_freq = np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)
    num_imfs = np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)
    psr = np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)
    mean_power = np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)
    centroid = np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)
    var_freq = np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)
    skew_freq =  np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)
    kurt_freq =  np.zeros((num_electrodes, length, features_per_block), dtype=np.float32)

    for i in range(num_electrodes):
        electrode = i+1

        # reload Numpy file of each electrode - features will be separated instead of electrodes.
        curr_path = os.path.join(f'{database}', 'HHT', 'temp', f'electrode{electrode}_hht.npy')
        combined = np.load(curr_path)

        # store features in separate arrays for now
        num_imfs[i] = combined[:, :, 0]
        peak_freq[i] = combined[:, :, 1]
        mean_power[i] = combined[:, :, 2]
        centroid[i] = combined[:, :, 3]
        var_freq[i] = combined[:, :, 4]
        skew_freq[i] = combined[:, :, 5]
        kurt_freq[i] = combined[:, :, 6]
        psr[i] = combined[:, :, 7]

        # remove temporary Numpy file
        os.remove(curr_path)

    # create h5 file with all electrodes and keys corresponding to the feature
    final_path = os.path.join(f'{database}', 'HHT', f'{iteration}_{exercise_num}_S{subject_num}_hht.h5')
    with h5py.File(final_path, 'w') as f:
        f.create_dataset('num imfs', data=num_imfs, compression='gzip', compression_opts=8)
        f.create_dataset('peak freq', data=peak_freq, compression='gzip', compression_opts=8)
        f.create_dataset('mean freq', data=centroid, compression='gzip', compression_opts=8)
        f.create_dataset('psr', data=psr, compression='gzip', compression_opts=8)
        f.create_dataset('mean power', data=mean_power, compression='gzip', compression_opts=8)
        f.create_dataset('skew freq', data=skew_freq, compression='gzip', compression_opts=8)
        f.create_dataset('kurt freq', data=kurt_freq, compression='gzip', compression_opts=8)
        f.create_dataset('var freq', data=var_freq, compression='gzip', compression_opts=8)

# CHANGE ACCORDING TO DATASET
database = 'DB4'
data_dict = {
        'DB1': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 100, 'electrodes': 10, 'subjects': 27, 'train': [1, 3, 4, 6, 7, 8, 9], 'test': [2, 5, 10], 'window length': 20, 'step': 1},
        'DB3': {'E1': 18, 'E2': 24, 'E3': 10, 'fs': 200, 'electrodes': 12, 'subjects': 11, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB4': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 12, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB5': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 16, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2}

        }

# global variables to be used
num_electrodes = data_dict[database]['electrodes']
num_subjects = data_dict[database]['subjects']
W = data_dict[database]['window length']
step = data_dict[database]['step']
 
train_trials = data_dict[database]['train']
test_trials = data_dict[database]['test']

if __name__ == '__main__':

    print("Number of cores: ", mp.cpu_count())

    for exercise in ['E1', 'E2', 'E3']:

        num_gestures = data_dict[database][exercise]
 
        for subject_num in range(1, num_subjects+1):

            # only DB3, DB4 and DB5 use this script, so file name is fine
            emg_path = os.path.join(f'{database}', 'Electrode Data', f'S{subject_num}_{exercise}_A1.mat')
            file = sio.loadmat(emg_path)
            labels = file['restimulus'].flatten()
            trials = file['rerepetition'].flatten()

            # if DB3 or DB4, then downsample from original 2000 Hz to 200 Hz
            if database in ['DB3', 'DB4']:
                downsample_factor = 10
                labels = labels[::downsample_factor]
                trials = trials[::downsample_factor]

            # split the labels according to the trial number - this creates a reproducible train/split 
            trial_split_index = helper.split_trials(trials, num_gestures, train_trials, test_trials)
            train_labels, test_labels = helper.split_labels(trial_split_index, labels, train_trials, test_trials)

            # all datasets that use this file are sampled at 200 Hz (or downsampled), hence 200 frequency bins from HHT
            num_bins = 200

            # create empty arrays to hold power spectrogram for each electrode
            train_hht = np.zeros((num_electrodes, len(train_labels), num_bins), dtype=np.float32)
            test_hht = np.zeros((num_electrodes, len(test_labels), num_bins), dtype=np.float32)

            # create directory if it doesn't exist
            os.makedirs(os.path.join(f'{database}', 'HHT'), exist_ok=True)

            # iterate through the spectrogram file, where each electrode is saved under an h5 file key to create one big 3D matrix with shape (electrode, t, freq_bins) that corresponds to train/test split
            file_path = os.path.join(f'{database}', 'Final', f'S{subject_num}_{exercise}_FREQ-TIME.h5')
            count = 0
            with h5py.File(file_path, 'r') as f:
                for name in f.keys():

                    curr_data = f[name]
                    # split the spectrogram into train/test and save in 3D array
                    train_hht[count], test_hht[count] = helper.split_data(trial_split_index, curr_data, len(train_labels), len(test_labels), train_trials, test_trials, num_bins)
                    count += 1

            # delete variables that are no longer needed - this has to happen, as I had memory issues on my laptop (16 GB RAM)
            del labels
            del trials
            del train_labels
            del test_labels

            # garbage collect immediately
            gc.collect()

           # multithreading - create 1 thread for each electrode
            process = []
            manager = mp.Manager()
            mutex = manager.Lock()
 
            for electrode, data in enumerate(train_hht):

                # feature extraction for each electrode
                thread = mp.Process(target=feature_extraction, args=(electrode+1, data, num_bins, 'Train', subject_num, exercise, mutex))
                process.append(thread)

                thread.start()
                # do not create all threads at the same time - memory issues!! Stagger them. 
                time.sleep(20)

            for p in process:
                p.join()
        
            # work out number of windows
            len_train_arr = (train_hht.shape[1] - W) // step + 1

            # call function to take each electrode feature information stored temporarily in memory and create dataset with all electrodes and all features
            create_dataset('Training', len_train_arr, num_electrodes, exercise, subject_num)
            del train_hht
            gc.collect()


            process_two = []
            mutex2 = manager.Lock()

            # multi-threading to create test dataset for each electrode
            for electrode, data in enumerate(test_hht):

                thread = mp.Process(target=feature_extraction, args=(electrode+1, data, num_bins, 'Test', subject_num, exercise, mutex2))
                process_two.append(thread)

                thread.start()
                # delay starting all threads to prevent memory issues
                time.sleep(20)

            for p in process_two:
                p.join()

            # create dataset for test data
            len_test_arr = (test_hht.shape[1] - W) // step + 1
            create_dataset('Test', len_test_arr, num_electrodes, exercise, subject_num)
