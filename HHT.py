import numpy as np
from PyEMD import EMD, CEEMDAN
import scipy.io as sio
import pandas as pd
import time
import multiprocessing as mp
import os
from scipy.signal import oaconvolve
import scipy.sparse as sp
import scipy.signal as signal
import gc
import h5py

def imf_decomposition(signal, t, electrode_name, electrode_num, send_end_time, subject_num, exercise, mutex):

    """
    Decompose the signal into the Hilbert-Huang (HHT) spectrum for each electrode.
    This function is multi-threaded - each electrode calls it.
    It takes a very long time (days to run E1-E3 for each dataset).

    Parameters:
        Signal (np.ndarray): The time-domain EMG signal of one electrode.
        t (np.ndarray): t array corresponding to length and sampling rate of signal.
        Electrode number and name: To print to terminal what electrode.
        Send End Time: Pipe to send the time back to the main function.
        Subject Number and Exercise: Current subject and their exercise being worked on.
        Mutex: Used when saving the HHT file to hard-disk.

    Returns:
        File: Saves HHT corresponding to the electrode to hard disk.
        Float: Total time taken to compute HHT for that electrode.
    """

    len_signal = len(signal)

    # split dataset with window size and step size
    window_len = data_dict[database]['window length']
    step_size = data_dict[database]['step']

    # make array of correct size - numpy array most efficient
    num_windows = int(np.floor( (len_signal-window_len) / step_size))

    # all windows including last one 
    windowed_t = np.empty((num_windows, window_len), dtype=np.float32)
    windowed_signal = np.empty((num_windows, window_len), dtype=np.float32)

    curr_pos = 0
    for i in range(num_windows):
        windowed_t[i] = t[curr_pos:(curr_pos+window_len)]
        windowed_signal[i] = signal[curr_pos:(curr_pos+window_len)]

        curr_pos += step_size

    # account for remaining signal values, where window size is too big; however, will use different vector so that numpy array has uniform sized vectors i.e. 40
    windowed_t_last = t[curr_pos:-1]
    windowed_signal_last = signal[curr_pos:-1]

    #typecast to float 16, akima to speed up computation
    ceemdan = CEEMDAN(DTYPE=np.float32, spline_kind='akima')

    # make dictionary for transformed data - needed because the 'depth' i.e. the number of IMFs per window is not known at creation
    # 'window 0' is first window up to 'window N-1'
    imf_dict = {f'window {i}': None for i in range(0, num_windows)}
    num_imfs_per_window = [None for i in range(0, num_windows+1)]

    t1 = time.time()

    # Perform CEEMDAN
    for index, key in enumerate(imf_dict):
        imf_dict[key] = ceemdan(windowed_signal[index])
        num_imfs_per_window[index] = len(imf_dict[key])
        print("S" + str(subject_num) + ", " + electrode_name + ", CEEMDAN: " + str(index+1) + '/' + str(num_windows+1) + ", IMFs: " + str(num_imfs_per_window[index]))

    imf_dict[f'window {num_windows}'] = ceemdan(windowed_signal_last)
    num_imfs_per_window[-1] = len(imf_dict[f'window {num_windows}'])

    # update variable afterwards, as last window will be called 'window N-1', where N is number of windows - due to zero indexing
    num_windows += 1

    # display in for for last IMF
    print("S" + str(subject_num) + ", " + electrode_name + ", CEEMDAN: " + str(num_windows) + '/' + str(num_windows) + ", IMFs: " + str(num_imfs_per_window[-1]), flush=True)

    tot_num_imfs = max(num_imfs_per_window)

    # 2D array to store (num imfs, value), once IMFs have been stitched back together after being calculated by windowing
    avg_imfs = np.zeros((tot_num_imfs, len_signal), dtype=np.float32)
    N_index = avg_imfs.copy()

    # iterate through IMFs and stitch IMFs together to create IMFs the length of the original signal
    for index, (key, imfs) in enumerate(imf_dict.items()):

        if index == (num_windows-1):
            t_array = windowed_t_last.copy()
        else:
            t_array = windowed_t[index]

        previous_t_index = None
        for i in range(len(imfs)):
            for j in range(len(imfs[i])):
                t_index = int(t_array[j] * data_dict[database]['fs'])                             # multiply time by sampling rate to get time index

                # check if the current t_index is the same as the previous one - if it is add one to increase
                if previous_t_index is not None and t_index == previous_t_index:
                    t_index += 1

                previous_t_index = t_index  
                avg_imfs[i][t_index] += imfs[i][j] 
                N_index[i][t_index] += 1

    for i in range(len(avg_imfs)):
        for t_index in range(len(avg_imfs[i])):                                                     # iterate over all time indices
            if N_index[i][t_index] != 0:                                                            # prevent division by zero
                avg_imfs[i][t_index] /= N_index[i][t_index]

    # IMPORTANT TO CHANGE DEPENDING ON DB - computes Hilbert Transform
    # scale frequencies, as index to sparse array have to be integers, not floats
    fs, coeff_num, freq_scale_factor = data_dict[database]['fs'], (data_dict[database]['window length']+1), 1e6

    mutex.acquire()                                                                                 # one thread at a time to access critical section, because the matrices will be big! Prevents memory issues
    print(electrode_name + " ACQUIRED MUTEX.")

    # call Hilbert-Huang transform function on the IMFs
    power, frequencies = HT(avg_imfs[1:-1], fs, len_signal, coeff_num, t)                           # sparse matrix returned from function - rows: time, columns: frequency. Don't use residual or first IMF?

    os.makedirs(os.path.join(f'{database}', 'Final', 'temp'), exist_ok=True)                        # if folder doesn't exist, then create it
    power_path = os.path.join(f'{database}', 'Final', 'temp', f'{electrode_num}_temp.npz')
    sp.save_npz(power_path, power)                                                                  # store to hard drive to free memory

    print(electrode_name + " RELEASED MUTEX.")
    mutex.release()                                                                                 # release mutex, once file has been saved

    del avg_imfs
    del power  
    gc.collect()                                                                                    # delete and garbage collect immediately to prevent memory issues
    
    t2 = time.time()
    send_end_time.send(t2-t1)                                                                       # send computation time through the pipe to be read in the main function

def FIRcoefficient(N):

    # N is the number of coefficients 
    # creates the FIR filter

    t = np.arange(-(N//2), (N//2)+1, 1)                                     # for the maths to work, t has to be non-causal
    h = np.zeros(N)                                                         # define h as filter coefficient array

    for index, n in enumerate(t):
        if n == 0: 
            h[index] = 0
        elif n % 2 != 0:
            h[index] = (2 / (np.pi*n))                                      # for n even, there is a value (given by equation); for n odd, leave at zero

    return h * np.hanning(N)

def backward_difference(x, h):
    
    # with x as input vector, and h the sampling period
    # simple approximation for derivative

    y = np.zeros(len(x))
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = (x[i] - x[i-1]) / h

    return y

def HT(imfs, fs, N, filter_length, t):

    """
    Takes IMFs and computes the Hilbert-Huang transform to return the power spectrogram of the signal.
    
    Parameters:
        IMFs (2D array): List of all computed IMFs - IMF number as first dimension, followed by the values which have the same length as original signal
        fs: Sampling rate.
        Filter length: This is given to the FIR function to create an arbitrary sized filter.
        t: Vector of sample times.

    Returns:
        Power Spectrogram (2D sparse matrix): Power spectrum of signal with shape (t, freq) - since frequency computed for each sample. All IMFs added to one spectrogram.
        Frequency bins (1D vector): The corresponding frequency bins, although this can be worked out, so it won't be used when returned.
    """

    # create FIR Hilbert filter
    hilbert_filter = FIRcoefficient(filter_length)

    # number of bins according to Huang
    T = N / fs
    num_freq_bins = fs 

    # maximum frequency that be represented due to sampling Nyquist = fs/2
    max_freq = fs / 2
    delta_f = max_freq / num_freq_bins

    # define frequency bins - including last frequency bin
    freq_bins = np.linspace(0, max_freq, num_freq_bins, endpoint=True)

    # s is the analytical signal - arrays to store magnitude and the index of the frequency bin
    s_freq_index = np.zeros(imfs.shape)
    s_mag = np.zeros(imfs.shape)

    # iterate through IMFs to convolve with Hilbert filter
    for index, x in enumerate(imfs):
        # oaconvolve used here, because Hilbert filter's length is much smaller than data length - apparently this convolution is quicker
        x_hat = oaconvolve(x, hilbert_filter, mode='same')

        # for instantaneous freq and phase. arctan2 to account for quadrant & need to unwrap phase!
        instant_phase = np.unwrap(np.arctan2(x_hat, x))
        instant_freq = backward_difference(instant_phase, 1/fs)  / (2 * np.pi)

        # from instantaneous frequency find the closest frequency bin. If the instantaneous frequency is less than zero, then it is in frequency bin zero; if inst. frequency is larger than max. frequency bin, then put it in max. frequency bin
        s_freq_index[index] = np.where(np.round(instant_freq / delta_f) < 0, 0,
                                    np.where(np.round(instant_freq / delta_f) > (num_freq_bins - 1), (num_freq_bins-1), 
                                    np.round(instant_freq / delta_f)))

        # for magnitude (energy as it's squared) of analytical signal at each time period
        s_mag[index] = (x_hat**2) + (x**2)

    scalogram = np.zeros((num_freq_bins, N), dtype=np.float32)

    # store all IMF data in the same spectrogram
    for imf_num in range((len(imfs))):
        for time in range(len(s_mag[imf_num])):

            # get frequency index
            freq_index = int(s_freq_index[imf_num][time])

            # an array to store magnitude (power) at frequency f and time t: time-frequency resolution. Power is the value, time is the row and frequency is the column
            scalogram[freq_index][time] += s_mag[imf_num][time]


    # convert matrix to sparse format - only stores non-zero elements with t values as row index & freq values as column index
    sparse_scalogram = sp.csr_matrix(scalogram)

    # explicitly delete matrix to ensure it's deleted, because it is huge and needs to be deallocated to avoid memory issues
    del scalogram
    gc.collect()

    return sparse_scalogram.T, freq_bins

# global variables to be used in functions e.g. window length
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

    # iterate through every exercise present in the dataset
    for exercise in ['E1', 'E2', 'E3']:

        # number of gestures changes depending on the exercise and dataset
        num_gestures = data_dict[database][exercise]
 
        for subject_num in range(1, num_subjects+1):

            t_start = time.time()

            # load EMG data - DB1 has different file name compared to DB3, DB4 and DB5
            if database == 'DB1':
                 emg_path = os.path.join(f'{database}', 'Electrode Data', f'S{subject_num}_A1_{exercise}.mat')
            else:
                emg_path = os.path.join(f'{database}', 'Electrode Data', f'S{subject_num}_{exercise}_A1.mat')

            # extract EMG data, the gesture or 'label' and trial number (not used in this code)
            file = sio.loadmat(emg_path)
            emg_data = file['emg'].T
            labels = file['restimulus'].flatten()
            trials = file['rerepetition'].flatten()

            # down sample 2k to 200 Hz for DB3 and DB4
            if database in ['DB3', 'DB4']:
                downsample_factor = 10
            
                # apply Butterworth low-pass filter to EMG data before downsampling
                cutoff_freq = fs/2 - 1
                order = 5
                original_fs = 2000
                b, a = signal.butter(order, cutoff_freq / (original_fs / 2), btype='low')
                signals_filtered = np.zeros_like(emg_data)
                for i in range(emg_data.shape[0]):
                    signals_filtered[i, :] = signal.filtfilt(b, a, emg_data[i, :])

                # now downsample
                emg_data = signals_filtered[:, ::downsample_factor]
                labels = labels[::downsample_factor]
                trials = trials[::downsample_factor]

            len_signal = emg_data.shape[1]                                  # all electrode signals should be same length
            t = np.arange(0, len_signal, 1)                                 # create time variable - all electrodes have same vector length
            t = t / data_dict[database]['fs']                               # adjust for the sampling frequency of 200 Hz

            # define 2 pipes for sending IMF results as a data frame, time it took to compute and info like no. of windows
            result_pipe = []
            time_pipe = []

            # multithreading - create 1 thread for each electrode
            process = []

            manager = mp.Manager()
            mutex = manager.Lock()
            for index, imf_data in enumerate(emg_data):
                electrode_name = f'Electrode {index+1}'

                # define pipe objects for data to be passed from threads to main function
                recv_end_time, send_end_time = mp.Pipe(False)

                # call function IMF decomposition which will save power spectrum to hard disk
                thread = mp.Process(target=imf_decomposition, args=(imf_data, t, electrode_name, index, send_end_time, subject_num, exercise, mutex))
                process.append(thread)

                # append sent time results to array local to main
                time_pipe.append(recv_end_time)

                thread.start()

            # copy time results from pipe to an array - do this before join() otherwise program hangs indefinitely
            ceemdan_time = [i.recv() for i in time_pipe]

            for p in process:
                p.join()

            # load HHT for each electrode back up from hard disk to save in a combined file
            file_path = os.path.join(f'{database}', 'Final', f'S{subject_num}_{exercise}_FREQ-TIME.h5')
            with h5py.File(file_path, 'w') as f:
                for i in range(num_electrodes):
                    electrode_file = os.path.join(f'{database}', 'Final', 'temp', f'{i}_temp.npz')
                    electrode_matrix = sp.load_npz(electrode_file)
                    f.create_dataset(f'Electrode_{i}', data=electrode_matrix.toarray(), compression='gzip', compression_opts=8)
                    os.remove(electrode_file)

            # print CEEMDAN time information
            print("S" + str(subject_num) + ": Time for CEEMDANs: " + str(ceemdan_time))
            print("S" + str(subject_num) + ": Max. CEEMDAN time: " + str(max(ceemdan_time)) +  "; and average CEEMDAN time: " + str(sum(ceemdan_time)/(len(ceemdan_time))))

            # print overall duration - the decomposition takes ages
            t_end = time.time()
            print("Time for whole computation of S" + str(subject_num) + " " + exercise + ": " + str(t_end-t_start))
       
