# Third-Year-Project-Code

The code can be divided into two main sections: the preprocessing scripts and the machine learning (ML) files. 

Preprocessing Files:
The three parts of this are the Hilbert-Huang transform (HHT), Short-time Fourier Transform (STFT) and the feature extraction files. All the files assume the original .mat files have been downloaded from https://ninapro.hevs.ch/. Further, it is assumed these Python files exist in the parent folder and the electrode data are in subdirectories in database file names e.g. /DB3/Electrode Data or /DB4/Electrode Data

train_test_split.py - a file with helper functions that are used by the preprocessing files

HHT.py - this is a multi-threaded file to decompose the signal into the HHT spectrum. It takes a very long time (matter of days for E1-E3).

STFT_feature_extraction.py - multi-threaded file to decompose the signal into 7 freuqency features. Can only be used for DB3, DB4 and DB5.
STFT_DB1_feature_extraction.py - same as above, but it can only be used for DB1 - the way the trials are split for train/test are different, as DB1 is split in CNN/SVM files, while the rest are split in feature extraction

feature_extraction.py - used to extract 11 time domain features from the EMG signal. Only used for DB3, DB4 and DB5.
DB1_feature_extraction.py - same as above, except only used for DB1 because the train/test data is split differently.

ML Files:
Two ML models were evaluated - SVMs and CNNs, but there are 4 files since DB1 is treated differently.

CNN.ipynb - assumes the features are saved on your google drive. Used for CNN and there are different feature vectors commented out.
CNN_DB1.ipynb - same as the above but it is only used for DB1.

SVM.ipynb - code for SVM. Only used for DB3, DB4 and DB5. Different feature vectors are commented out and can be experimented with.
SVM_DB1.ipynb - same as above, except it is only implemented for DB1.
