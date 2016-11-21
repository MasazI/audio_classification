#encoding: utf-8
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wave
from matplotlib.pyplot import specgram
from task import DEBUG_DIR


def print_wave_info(file):
    wf = wave.open(file, "r")
    print "チャンネル数:", wf.getnchannels()
    print "サンプル幅:", wf.getsampwidth()
    print "サンプリング周波数:", wf.getframerate()
    print "フレーム数:", wf.getnframes()
    print "パラメータ:", wf.getparams()
    print "長さ（秒）:", float(wf.getnframes()) / wf.getframerate()


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(3, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.985, fontsize=18)
    #plt.show()
    plt.savefig(os.path.join(DEBUG_DIR, 'f1_waveplot.jpg'))


def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(3, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.985, fontsize=18)
    #plt.show()
    plt.savefig(os.path.join(DEBUG_DIR, 'f2_spec.jpg'))


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(3, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram', x=0.5, y=0.985, fontsize=18)
    #plt.show()
    plt.savefig(os.path.join(DEBUG_DIR, 'f3_logpowerspec.jpg'))