#encoding: utf-8
import os
import librosa
import numpy as np
import glob
import util

TRAIN_DATA_RATIO = 10

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


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode