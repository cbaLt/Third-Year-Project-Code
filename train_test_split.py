import numpy as np

def rolling_window_HHT_spectrum(arr, window_len, step):

    """
    Segment 2D 'continuous' spectrogram into windows to create 2D matrix.

    Parameters:
        arr (2D array): The array to segment.
        Window Length (int): Window Size - depending on dataset.
        Step size (int): Step / stride size

    Returns:
        Windowed arr (3D matrix): Same array but now 3D, as it has been segmented
    """

    num_windows = (arr.shape[0] - window_len) // step + 1
    windows = np.zeros((num_windows, window_len, arr.shape[1]), dtype=np.float32)

    for i in range(num_windows):
      start = i * step
      end = start + window_len
      windows[i] = arr[start:end, :]  # slice along columns

    return windows

def split_trials(trial_data, num_gestures, training_trials, test_trials):

    """
    Calculates indexes corresponding to each trial which will be used to split all data into train/test arrays. Very specific to DB3, DB4 and DB5.

    Parameters:
        Trial data (1D vector): Data from Ninapro with each integer corresponding to the trial number.
        Number of gestures (int): Number of gestures which is dependent on the exercise and dataset.
        Training and test trials (both 1D vectors): Both small arrays which list what trials will be used for training or testing e.g. 6 trials for DB5, four will be used for training, two for testing
 
    Returns:
        Trial Index array (3D array): Indices corresponding to where in the signal that gesture and that trial occurred. Each gesture is performed a set number of times or 'trials'.
    """

    # calculate number of trials from arrays given. Create 3D array to store indices for each trial, for each gesture
    num_trials = len(training_trials) + len(test_trials)
    trial_index = np.zeros((num_trials, num_gestures, 2))

    curr_label = 0                                     # label indexed from 0, so gesture 1 is label 0 etc.
    for i in range(1, len(trial_data)):
        curr_trial = trial_data[i] - 1              # minus 1 to make zero indexed - trial 1 is 0 etc.
        prev_trial = trial_data[i-1] - 1

        if curr_trial != prev_trial:                   # store index at beginning and end of trial

            # zero indexed
            if curr_trial == 0:                                 # if trial is zero (trial 1), that means it is a new gesture (label)
                trial_index[prev_trial][curr_label][1] = i      # store index for beginning of that trial, for that gesture
                curr_label += 1                                 # update gesture (label) by 1

                trial_index[curr_trial][curr_label][0] = i      # store index for end of that trial - it is an index for range() function, so this value will not be included, e.g. 2017, means up to 2016

            else:
                trial_index[prev_trial][curr_label][1] = i      # same as before, except there is no need to update gesture (label), as trial is between 1-6 (or 0-5 zero-indexed)
                trial_index[curr_trial][curr_label][0] = i

    # update last trial, last gesture (label) beginning index, as it is wrong!!
    trial_index[(num_trials-1)][(num_gestures-1)][0] = trial_index[(num_trials-2)][(num_gestures-1)][1]

    return trial_index

def split_data(index_info, arr, train_len, test_len, training_trials, test_trials, num_bins):

    """
    Takes the test/train split indices and splits the HHT spectrogram accordingly.

    Parameters:
        Index Info (3D array): Array that was returned by split_trial function, which holds the indices to split the train/test data.
        Arr (2D array): The spectrogram from one electrode, which is to be split into train/test data.
        Test / train len (both ints): Integers corresponding to the resulting lengths of the train and test arrays.
        Training and test trials (both 1D vectors): Both small arrays which list what trials will be used for training or testing e.g. 6 trials for DB5, four will be used for training, two for testing
        Num bins (int): Number of frequency bins.
 
    Returns:
        Train (2D array): Subset of original spectrogram, which will be used for training of ML model.
        Test (2D array): Subset of original spectrogram to be used for testing the ML model.
    """

    train = np.empty((0, num_bins), dtype=np.float32)
    test = np.empty((0, num_bins), dtype=np.float32)

    # use the indices found to create arrays for training data
    for j in training_trials:
        trial_num = j-1

        for indices in index_info[trial_num]:
            beg, end = int(indices[0]), int(indices[1])

            # slice the training labels and features
            train = np.append(train, arr[beg:end, :], axis=0)


    # use the indices found to slice testing features and labels
    for i in test_trials:
        trial_num = i-1

        for indices in index_info[trial_num]:

            beg, end = int(indices[0]), int(indices[1])
            test = np.append(test, arr[beg:end, :], axis=0)

    # return Numpy arrays of spectrogram
    return np.array(train), np.array(test)

def split_labels(index_info, label_data, training_trials, test_trials):

    """
    Calculates indexes corresponding to each trial which will be used to split all data into train/test arrays. Very specific to DB3, DB4 and DB5.

    Parameters:
        Index Info (3D array): Array that was returned by split_trial function, which holds the indices to split the train/test data.
        Label data (1D array): Label data to be split into train/split sets.
        Training and test trials (both 1D vectors): Both small arrays which list what trials will be used for training or testing e.g. 6 trials for DB5, four will be used for training, two for testing
 
    Returns:
        Train (1D array): Subset of original labels, which will be used for training of ML model.
        Test (1D array): Subset of original labels to be used for testing the ML model.
    """

    training_labels, test_labels = [], []

    # use the indices found to create arrays for training data (labels and features)
    for j in training_trials:
        trial_num = j-1

        for indices in index_info[trial_num]:
                beg, end = int(indices[0]), int(indices[1])

                # slice the training labels and features
                training_labels = np.append(training_labels, label_data[beg:end])

    # use the indices found to slice testing features and labels
    for j in test_trials:
        trial_num = j-1

        for indices in index_info[trial_num]:

                beg, end = int(indices[0]), int(indices[1])
                test_labels = np.append(test_labels, label_data[beg:end])

    return training_labels, test_labels


def split_time_series(index_info, time_series_arr, train_len, test_len, training_trials, test_trials):
    """
    Splits raw EMG data into train / test split, so that the feature extraction can be separate

    Parameters:
        Index Info: Generated from split trials function to determine indices for train/test split.
        Time Series Arr (2D array): First dimension is the electorde, second is the EMG data.
        Train / Test Length: The resulting array lengths.
        Training / Test Trials: Small arrays contain the trial numbers used for train/test split.
 
    Returns:
        Train Series (2D array): First dimension is electrode, second is the EMG data which is a subset of orginal data.
        Test Series (2D array):  First dimension is electrode, second is the EMG data which is a subset of orginal data for testing ML model.

    """

    electrode_num = time_series_arr.shape[0]
    # create empty arrays
    train_split_series = np.zeros((electrode_num, train_len), dtype=np.float32)
    test_split_series = np.zeros((electrode_num, test_len), dtype=np.float32)

    for num in range(electrode_num):

      training_series, test_series= [], []

      # use the indices found to create arrays for training data
      for j in training_trials:
          trial_num = j-1

          for indices in index_info[trial_num]:
              beg, end = int(indices[0]), int(indices[1])

              # slice the training labels and features
              training_series = np.append(training_series, time_series_arr[num, beg:end])

      train_split_series[num] = training_series

      # use the indices found to slice testing features and labels
      for i in test_trials:
          trial_num = i-1

          for indices in index_info[trial_num]:

              beg, end = int(indices[0]), int(indices[1])
              test_series = np.append(test_series, time_series_arr[num, beg:end])


      test_split_series[num] = test_series

    # return the subset of original EMG data
    return train_split_series, test_split_series

def rolling_window_2d(arr, window_len, step):

    # used when to segement 2D array e.g. EMG data with first dimension as electrodes, second as data
    num_windows = (arr.shape[1] - window_len) // step + 1
    windows = np.zeros((arr.shape[0], num_windows, window_len), dtype=arr.dtype)

    for i in range(num_windows):
      start = i * step
      end = start + window_len
      windows[:, i] = arr[:, start:end]  # Slice along columns

    return windows

def rolling_window(arr, window_len, step):

    num_windows = (len(arr) - window_len) // step + 1
    windows = np.zeros((num_windows, window_len), dtype=arr.dtype)

    for i in range(num_windows):
        start = i * step
        end = start + window_len
        windows[i] = arr[start:end]

    return windows
