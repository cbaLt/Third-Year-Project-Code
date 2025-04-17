import numpy as np
import scipy.io as sio
import multiprocessing as mp
import os
import scipy.sparse as sp
import gc
import h5py
import statsmodels.api as sm
from scipy.fft import fft
import scipy.signal as signal
import time
# custom functions to split train/test data for DB3, DB4 and DB5
import train_test_split as helper

np.set_printoptions(threshold=np.inf)

def SSC(x, threshold):

    # computes slope sign change

    N = len(x)
    ssc = 0  # Initialize SSC sum
    
    for i in range(1, N - 1):  # Loop through from the second to the second last element
        # Calculate the difference between consecutive samples
        diff1 = np.abs(x[i] - x[i-1])
        diff2 = np.abs(x[i] - x[i+1])
        
        # Compute the product of the differences
        product = diff1 * diff2
        
        # Apply threshold function f(x)
        if x[i] > x[i-1] and x[i] > x[i+1]:
            if diff1 > threshold or diff2 > threshold:
                ssc += 1
        elif x[i] < x[i+1] and x[i] < x[i-1]:
            if diff1 > threshold or diff2 > threshold:
                ssc += 1
    return ssc


def feature_extraction(x, subject, electrode_num, iteration, threshold, mutex):

    # combined array in the following order: mav, mavs, Wilson Amplitude, zero crossings, variance, slope sign change and 4 autoregressive coefficients
    combined_arr = np.zeros((13, x.shape[0]), dtype=np.float32)
    
    # used to calculate MAVS
    prev_mav = 0
    num_windows = x.shape[0]
    window_len = x.shape[1]

    for window in range(x.shape[0]):
        print(f'S{subject}, Electrode {electrode_num} {iteration} Features: {window+1} / {num_windows}')

        arr = x[window]
        abs_x = np.abs(arr)
        mean_arr = np.mean(abs_x)

        # MAV
        combined_arr[0][window] = mean_arr

        if window != 0:
            # MAVS
            combined_arr[1][window-1] = mean_arr - prev_mav

        prev_mav = mean_arr

        # waveform length
        combined_arr[2][window] = np.sum(np.abs(np.diff(arr)))

        # Wilson amplitude
        wap = np.abs(arr[1:] - arr[:-1])
        combined_arr[3][window] = sum(1 for diff in wap if diff > threshold)

        # Zero Crossing rate
        sign_change = np.sign(arr)
        cond = [1 for j in range(1, len(arr)) if sign_change[j]+sign_change[j-1] == 0 and np.abs(arr[j-1] - arr[j]) > threshold]
        combined_arr[4][window] = np.sum(cond)

        # AR4
        AR_model = sm.tsa.AutoReg(x[window], lags=4)
        ar_fit = AR_model.fit()
        ar_coeff = ar_fit.params
        combined_arr[5][window] = ar_coeff[0]
        combined_arr[6][window] = ar_coeff[1]
        combined_arr[7][window] = ar_coeff[2]
        combined_arr[8][window] = ar_coeff[3]

        # RMS
        combined_arr[9][window] = np.sqrt(np.sum(arr**2)) / window_len

        # slope sign change
        combined_arr[10][window] = SSC(arr, threshold)

        # VAR
        combined_arr[11][window] = np.sum(arr**2) / (window_len-1)

        # IEMG
        combined_arr[12][window] = np.sum(np.abs(arr))

    mutex.acquire()
    print(f'Electrode {electrode_num} Acquired Mutex.')
    os.makedirs(os.path.join(f'{database}', 'Features_down', 'temp'), exist_ok=True)
    path = os.path.join(f'{database}', 'Features_down', 'temp', f'electrode{electrode_num}_features.npz')
    sp.save_npz(path, sp.csr_matrix(combined_arr))
    print(f'Electrode {electrode_num} Released Mutex.///////////////////////////////////////////////////////////////////////////////////')
    mutex.release()

def format_data(label, data, trials, num_getures, training_trials, test_trials):

    # used to make main function simpler - simply calls helper functions to format train/test split

    M = data_dict[database]['window length']
    step = data_dict[database]['step']

    # split and window data 
    trial_split_index = helper.split_trials(trials, num_gestures, training_trials, test_trials)
    train_labels, test_labels = helper.split_labels(trial_split_index, label, training_trials, test_trials)
    train_time_series, test_time_series = helper.split_time_series(trial_split_index, data, len(train_labels), len(test_labels), training_trials, test_trials)

    # window labels - only to predict whether gesture or not
    print(len(test_labels), len(train_labels))
    test_label_arr = helper.rolling_window(test_labels, M, step)
    train_label_arr = helper.rolling_window(train_labels, M, step)

    # format labels
    train_label_one = [np.max(arr) for arr in train_label_arr]
    test_label_one = [np.max(arr) for arr in test_label_arr]

    # window across all electrodes
    train_series = helper.rolling_window_2d(train_time_series, M, step)
    test_series = helper.rolling_window_2d(test_time_series, M, step)

    return train_series, test_series

def create_dataset(iteration, length, num_electrodes, exercise_num, subject_num):

    # individual arrays for each feature extracted
    MAV = np.zeros((num_electrodes, length), dtype=np.float32)
    MAVS = np.zeros((num_electrodes, length), dtype=np.float32)
    WAP = np.zeros((num_electrodes, length), dtype=np.float32)
    WL = np.zeros((num_electrodes, length), dtype=np.float32)
    ZC = np.zeros((num_electrodes, length), dtype=np.float32)
    coeff1 = np.zeros((num_electrodes, length), dtype=np.float32)
    coeff2 = np.zeros((num_electrodes, length), dtype=np.float32)
    coeff3 = np.zeros((num_electrodes, length), dtype=np.float32)
    coeff4 = np.zeros((num_electrodes, length), dtype=np.float32)
    var = np.zeros((num_electrodes, length), dtype=np.float32)
    iemg = np.zeros((num_electrodes, length), dtype=np.float32)
    ssc = np.zeros((num_electrodes, length), dtype=np.float32)
    rms = np.zeros((num_electrodes, length), dtype=np.float32)

    for i in range(num_electrodes):
        electrode = i+1

        curr_path = os.path.join(f'{database}','Features_down', 'temp', f'electrode{electrode}_features.npz')
        sparse_matrix = sp.load_npz(curr_path)
        combined = sparse_matrix.toarray()

        # combined array in the following order: mav, mavs, Wilson Amplitude, zero crossings, variance, slope sign change and AR coefficients
        MAV[i] = combined[0]
        MAVS[i] = combined[1]
        WL[i] = combined[2]
        WAP[i] = combined[3]
        ZC[i] = combined[4]
        coeff1[i] = combined[5]
        coeff2[i] = combined[6]
        coeff3[i] = combined[7]
        coeff4[i] = combined[8]
        rms[i] = combined[9]
        ssc[i] = combined[10]
        var[i] = combined[11]
        iemg[i] = combined[12]

        os.remove(curr_path)

    # create new file with features as keys
    final_path = os.path.join(f'{database}', 'Features_down', f'{iteration}_{exercise_num}_S{subject_num}_features.h5')
    with h5py.File(final_path, 'w') as f:
        f.create_dataset('MAV', data=MAV, compression='gzip', compression_opts=8)
        f.create_dataset('MAVS', data=MAVS, compression='gzip', compression_opts=8)
        f.create_dataset('WL', data=WL, compression='gzip', compression_opts=8)
        f.create_dataset('WAP', data=WAP, compression='gzip', compression_opts=8)
        f.create_dataset('ZC', data=ZC, compression='gzip', compression_opts=8)
        f.create_dataset('ar1', data=coeff1, compression='gzip', compression_opts=8)
        f.create_dataset('ar2', data=coeff2, compression='gzip', compression_opts=8)
        f.create_dataset('ar3', data=coeff3, compression='gzip', compression_opts=8)
        f.create_dataset('ar4', data=coeff4, compression='gzip', compression_opts=8)
        f.create_dataset('RMS', data=rms, compression='gzip', compression_opts=8)
        f.create_dataset('SSC', data=ssc, compression='gzip', compression_opts=8)
        f.create_dataset('VAR', data=var, compression='gzip', compression_opts=8)
        f.create_dataset('IEMG', data=iemg, compression='gzip', compression_opts=8)


database = 'DB4'
data_dict = {
        'DB1': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 100, 'electrodes': 10, 'subjects': 27, 'train': [1, 3, 4, 6, 7, 8, 9], 'test': [2, 5, 10], 'window length': 20, 'step': 1},
        'DB3': {'E1': 18, 'E2': 24, 'E3': 10, 'fs': 200, 'electrodes': 12, 'subjects': 11, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB4': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 12, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2},
        'DB5': {'E1': 13, 'E2': 18, 'E3': 24, 'fs': 200, 'electrodes': 16, 'subjects': 10, 'train': [1, 3, 4, 6], 'test': [2, 5], 'window length': 40, 'step': 2}

        }

num_electrodes = data_dict[database]['electrodes']
num_subjects = data_dict[database]['subjects']
fs = data_dict[database]['fs']
W = data_dict[database]['window length']

train_trials = data_dict[database]['train']
test_trials = data_dict[database]['test']

if __name__ == '__main__':

    print("Number of cores: ", mp.cpu_count())

    for exercise in ['E1', 'E2', 'E3']:
        num_gestures = data_dict[database][exercise]

        for subject_num in range(1, num_subjects+1):

            # for DB3, DB4 and DB5
            emg_path = os.path.join(f'{database}', 'Electrode Data', f'S{subject_num}_{exercise}_A1.mat')

            file = sio.loadmat(emg_path)
            emg_data = file['emg'].T
            labels = file['restimulus'].flatten()
            trials = file['rerepetition'].flatten()

            # down sample 2k to 100 Hz or 200 Hz
            if database in ['DB3', 'DB4']:

                downsample_factor = 10
                cutoff_freq = fs/2 - 1
                order = 5
                original_fs = 2000
                # low-pass butterworth filter with cut-off freq. 1 below Nyquist
                b, a = signal.butter(order, cutoff_freq / (original_fs / 2), btype='low')
                signals_filtered = np.zeros_like(emg_data)
                for i in range(emg_data.shape[0]):
                    signals_filtered[i, :] = signal.filtfilt(b, a, emg_data[i, :])

                emg_data = signals_filtered[:, ::downsample_factor]

                # down sample other signals
                labels = labels[::downsample_factor]
                trials = trials[::downsample_factor]

            rest_arr = []
            for index in range(len(labels)):
                if labels[index] == 0:
                    rest_arr.append(index)
            
            rest_electrodes = emg_data[:, rest_arr]
            rest_mean = np.mean(np.abs(np.diff(rest_electrodes, axis=1)), axis = 1)
            rest_std = np.std(np.abs(np.diff(rest_electrodes, axis=1)), axis = 1)
            electrode_threshold = rest_mean

            train_data, test_data = format_data(labels, emg_data, trials, num_gestures, train_trials, test_trials)

            # multithreading - create 1 thread for each electrode
            process = []

            manager = mp.Manager()
            mutex = manager.Lock()

            # multi-threading function - saves electrode feature files to hard disk
            for index, data in enumerate(train_data):
                # define pipe objects for data to be passed from threads to main function

                thread = mp.Process(target=feature_extraction, args=(data, subject_num, index+1, 'Train', electrode_threshold[index], mutex))
                process.append(thread)

                thread.start()

            for p in process:
                p.join()

            # reload electrode data to create single train file
            create_dataset('Training', train_data.shape[1], num_electrodes, exercise, subject_num)

            process_two = []
            mutex2 = manager.Lock()

            # multi-threading function - saves electrode feature files to hard disk
            for index, data in enumerate(test_data):

                thread = mp.Process(target=feature_extraction, args=(data, subject_num, index+1, 'Test', electrode_threshold[index], mutex2))
                process_two.append(thread)

                thread.start()

            for p in process_two:
                p.join()

            # reload electrode data to create single test file
            create_dataset('Test', test_data.shape[1], num_electrodes, exercise, subject_num)

