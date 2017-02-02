#encoding: utf-8
import os
import librosa
import numpy as np
import glob
import util
import matplotlib.pyplot as plt

TRAIN_DATA_RATIO = 10
WINDOW = 512

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    # sftf
    stft = np.abs(librosa.stft(X))
    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    return mfccs,chroma,mel,contrast

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_cnn_feature(file_name, label, bands=60, frames=41, verbose=False):
    window_size=WINDOW*(frames-1)
    log_specgrams = []
    labels = []
    X, sample_rate = librosa.load(file_name)
    for (start,end) in windows(X, window_size):
        if(len(X[start:end]) == window_size):
            signal = X[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
            if verbose:
                librosa.display.specshow(melspec, x_axis='time')
                plt.colorbar()
                plt.title('MELSPEC')
                plt.tight_layout()

            logspec = librosa.logamplitude(melspec)
            if verbose:
                librosa.display.specshow(logspec, x_axis='time')
                plt.colorbar()
                plt.title('LOGSPEC')
                plt.tight_layout()

            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
            labels.append(label)
    return log_specgrams, labels


def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    # file load
    labeld_dict = {}
    with open('all_labeled_data.csv', 'r') as f:
        for line in f:
            file_label = line.split(",")
            labeld_dict[file_label[0]] = int(file_label[1])
    features, labels = np.empty((0,187)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print("label: %s" % (label))
        print("sub_dir: %s" % (sub_dir))
        for i, fn in enumerate(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):
            if i % TRAIN_DATA_RATIO != 0:
                continue
            label = labeld_dict[fn]
            if label == 2:
                continue
            print("%d: extract file: %s" % (i, fn))
            # try:
            #     util.print_wave_info(fn)
            # except Exception as e:
            #     print("[Error] unhandle audio file. %s" % (e))
            #     continue
            try:
                mfccs, chroma, mel, contrast = extract_feature(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype = np.int)


def parse_audio_files_cnn(parent_dir,sub_dirs,bands=60,frames=41,file_ext='*.wav',verbose=False):
    # file load
    labeld_dict = {}
    with open('all_labeled_data.csv', 'r') as f:
        for line in f:
            file_label = line.split(",")
            labeld_dict[file_label[0]] = int(file_label[1])
    window_size = WINDOW * (frames - 1)
    log_specgrams = []
    labels = []
    fns = []

    for label, sub_dir in enumerate(sub_dirs):
        print("label: %s" % (label))
        print("sub_dir: %s" % (sub_dir))
        for i, fn in enumerate(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))):
            if i % TRAIN_DATA_RATIO != 0:
                continue
            label = labeld_dict[fn]
            if label == 2:
                continue
            print("%d: extract file: %s" % (i, fn))

            X, sample_rate = librosa.load(fn)
            for (start, end) in windows(X, window_size):
                if (len(X[start:end]) == window_size):
                    signal = X[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)

                    if verbose:
                        print("print")
                        librosa.display.specshow(melspec, x_axis='time')
                        plt.colorbar()
                        plt.title('MFCC')
                        plt.tight_layout()
                        plt.show()

                    logspec = librosa.logamplitude(melspec)
                    if verbose:
                        print("logspec shape: ", logspec.shape)
                        librosa.display.specshow(logspec, x_axis='time')
                        plt.colorbar()
                        plt.title('LOGSPEC')
                        plt.tight_layout()
                        plt.show()

                    delta = librosa.feature.delta(logspec, order=2)
                    if verbose:
                        librosa.display.specshow(delta, x_axis='time')
                        plt.colorbar()
                        plt.title('DELTA')
                        plt.tight_layout()
                        plt.show()

                    #logspec = logspec.T.flatten()[:, np.newaxis].T
                    # print("logspec flatten shape", logspec.shape)
                    log_specgrams.append(logspec)
                    labels.append(label)
                    fns.append(fn)
    print("log_specgrams shape: %s" % len(log_specgrams))
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    print("log_specgrams re-shape: ", log_specgrams.shape)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    print("features shape: ", features.shape)

    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        if verbose:
            logspec = features[i, :, :, 0]
            deltaspec = features[i, :, :, 1]
            print("label: %d" % labels[i])
            print("fn: %s" % fns[i])
            print("logspec shape in features: ", logspec.shape)
            librosa.display.specshow(logspec, x_axis='time')
            plt.colorbar()
            plt.title('logspec')
            plt.tight_layout()
            plt.show()
            librosa.display.specshow(deltaspec, x_axis='time')
            plt.colorbar()
            plt.title('deltaspec')
            plt.tight_layout()
            plt.show()

    features = np.array(features)
    labels = np.array(labels, dtype=np.int)
    tmp = np.concatenate((features.reshape(len(features), -1), labels.reshape(len(labels), -1)), axis=1)
    np.random.shuffle(tmp)

    features_shuffle = tmp[:, :features.size//len(features)].reshape(features.shape)
    labels_shuffle = tmp[:, features.size//len(features):].reshape(labels.shape)

    print("features_shuffle shape: ", features_shuffle.shape)
    print("labels_shuffle shape: ", labels_shuffle.shape)
    return features_shuffle, np.array(labels_shuffle, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode