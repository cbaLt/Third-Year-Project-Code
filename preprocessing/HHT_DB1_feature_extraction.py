import numpy as np
import scipy.io as sio
import pandas as pd
import multiprocessing as mp
import os
import scipy.sparse as sp
import h5py
import gc
import time

def feature_extraction(electrode_num, spectrum, freq_bins, subject, exercise, mutex):

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

    fs = data_dict[database]['fs']
    window_len = data_dict[database]['window length']

    freq_arr = np.linspace(0, fs/2, freq_bins)
    num_windows = (spectrum.shape[0] - window_len) // step + 1

    # this is currently set to one block per window segment - can be changed to extract more features from each segment with HHT
    block_time = 20
    block_freq = 100
    step_time = 1
    step_freq = 1

    # known before run-time, also listed in global data dictionary
    time_frames, freq_bins = 20, 100

    # Number of blocks
    num_blocks_time = (time_frames - block_time) // step_time + 1
    num_blocks_freq = (freq_bins - block_freq) // step_freq + 1

    # 8 features per block of spectrogram
    features = np.zeros((num_windows, (num_blocks_time*num_blocks_freq), 8), dtype=np.float32)

    # Create the windows by slicing the array along the first axis (rows)
    for i in range(num_windows):
        start = i * step
        end = start + window_len
        segment = spectrum[start:end, :]  # Slice along rows (first dimension)
 
        print(f'{exercise}, S{subject}; Electrode {electrode_num}: Segment {i+1}/{num_windows}')

        block_features = []
        for t in range(0, time_frames - block_time + 1, step_time):
            for f in range(0, freq_bins - block_freq + 1, step_freq):
                block = segment[t:t+block_time, f:f+block_freq]

                # IMF count
                count_imfs = np.count_nonzero(block)
                num_imfs = count_imfs / block.shape[0]

                # Max frequency (peak freq bin)
                peak_freq_bin = np.argmax(block)
                peak_freq_idx = np.unravel_index(peak_freq_bin, block.shape)
                peak_freq = freq_arr[f + peak_freq_idx[1]]  # offset by block starting freq

                # Mean power
                mean_power = np.sum(block) / block_freq

                # Spectral moments
                block_sum = np.sum(block)
                freq_slice = freq_arr[f:f+block_freq]

                if block_sum > 0:
                    # mean and variance are not standardised
                    centroid = np.sum(np.sum(block, axis=0) * freq_slice) / block_sum
                    var_freq = np.sum(np.sum(block, axis=0) * (freq_slice - centroid)**2) / block_sum

                    # skewness and kurtosis are standardised to variance
                    skew_freq = np.sum(np.sum(block, axis=0) * (freq_slice - centroid)**3) / (block_sum * var_freq**1.5 + 1e-12)
                    kurt_freq = np.sum(np.sum(block, axis=0) * (freq_slice - centroid)**4) / (block_sum * var_freq**2 + 1e-12)
                else:
                    centroid, var_freq, skew_freq, kurt_freq = 0, 0, 0, 0

                # Power Spectrum Ratio (PSR) - it is known the frequency resolution is 0.5 Hz
                freq_resolution = 0.5
                f_interest = int(5 / freq_resolution)

                # sum within frequency range across all time frames in the block
                local_start = max(peak_freq_idx[1] - f_interest, 0)
                local_end = min(peak_freq_idx[1] + f_interest + 1, block.shape[1])
                psr = np.sum(block[:, local_start:local_end]) / (np.sum(block) + 1e-12)

                block_features.append([
                    num_imfs, peak_freq, mean_power,
                    centroid, var_freq, skew_freq, kurt_freq,
                    psr
                ])

        features[i] = np.array(block_features, dtype=np.float32)

    # mutex for critical section when saving Numpy array to hard disk
    mutex.acquire()
    print(f'Electrode {electrode_num} Acquired Mutex.')
    os.makedirs(os.path.join(f'{database}', 'HHT', 'temp'), exist_ok=True)
    path = os.path.join(f'{database}', 'HHT', 'temp', f'electrode{electrode_num}_hht.npy')
    np.save(path, features)
    print(f'Electrode {electrode_num} Released Mutex.///////////////////////////////////////////////////////////////////////////////////')
    mutex.release()


def create_dataset(length, num_electrodes, exercise_num, subject_num):

    """
    It is just used to reload the electrode data saved to hard disk and put it into a single file.

    Parameters:
        All parameters are just for formatting.
 
    Returns:
        None: Saves one file to hard disk in subdirectory corresponding to the dataset.
    """

    # this number is changed depending on how many blocks were used in feature extraction function
    features_per_block = 1
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

        # store in features in separate arrays for now
        num_imfs[i] = combined[:, :, 0]
        peak_freq[i] = combined[:, :, 1]
        mean_power[i] = combined[:, :, 2]
        centroid[i] = combined[:, :, 3]
        var_freq[i] = combined[:, :, 4]
        skew_freq[i] = combined[:, :, 5]
        kurt_freq[i] = combined[:, :, 6]
        psr[i] = combined[:, :, 7]

        # remove temporary electrode file
        os.remove(curr_path)

    # create h5 file with all electrodes and keys corresponding to the feature
    final_path = os.path.join(f'{database}', 'HHT', f'{exercise_num}_S{subject_num}_hht.h5')
    with h5py.File(final_path, 'w') as f:
        f.create_dataset('num imfs', data=num_imfs, compression='gzip', compression_opts=8)
        f.create_dataset('peak freq', data=peak_freq, compression='gzip', compression_opts=8)
        f.create_dataset('mean freq', data=centroid, compression='gzip', compression_opts=8)
        f.create_dataset('psr', data=psr, compression='gzip', compression_opts=8)
        f.create_dataset('mean power', data=mean_power, compression='gzip', compression_opts=8)
        f.create_dataset('skew freq', data=skew_freq, compression='gzip', compression_opts=8)
        f.create_dataset('kurt freq', data=kurt_freq, compression='gzip', compression_opts=8)
        f.create_dataset('var freq', data=var_freq, compression='gzip', compression_opts=8)


# ONLY DB1 USES THIS FILE
database = 'DB1'
data_dict = {
        'DB1': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 100, 'electrodes': 10, 'subjects': 27, 'train': [1, 3, 4, 6, 7, 8, 9], 'test': [2, 5, 10], 'window length': 20, 'step': 1},
        'DB2': {'E1': 18, 'E2': 24, 'E3': 10, 'fs': 100, 'electrodes': 12, 'subjects': 40, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 400, 'step': 20},
        'DB3': {'E1': 18, 'E2': 24, 'E3': 10, 'fs': 200, 'electrodes': 12, 'subjects': 11, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB4': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 12, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB5': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 16, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2}

        }

# GLOBAL VARIABLES
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

            file = sio.loadmat(os.path.join(f'{database}', 'Electrode Data', f'S{subject_num}_A1_{exercise}.mat'))
            emg_data = file['emg'].T

            # 100 frequency bins for DB1
            num_bins = 100
            num_windows = (emg_data.shape[1] - W) // step + 1
            hht = np.zeros((num_electrodes, emg_data.shape[1], num_bins), dtype=np.float32)
            os.makedirs(os.path.join(f'{database}', 'HHT'), exist_ok=True)

            file_path = os.path.join(f'{database}', 'Final', f'S{subject_num}_{exercise}_FREQ-TIME.h5')
            # load HHT data and store in one large 3D array
            count = 0
            with h5py.File(file_path, 'r') as f:
                for name in f.keys():
                    curr_data = f[name]
                    hht[count, :, :] = curr_data
                    count += 1

            process = []
            manager = mp.Manager()
            mutex = manager.Lock()

            # multi-threading for each electrode
            for electrode, data in enumerate(hht):

                thread = mp.Process(target=feature_extraction, args=(electrode+1, data, num_bins, subject_num, exercise, mutex))
                process.append(thread)

                thread.start()
                # to prevent memory issues
                time.sleep(8)

            for p in process:
                p.join()
        
            # reload electrode data to input to one file
            create_dataset(num_windows, num_electrodes, exercise, subject_num)
            del hht
            gc.collect()
