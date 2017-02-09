#encoding: utf-8
import os
import librosa
import struct
import wave
import numpy as np
import scipy.signal
from pylab import *

OUTPUT_DIR = '/mnt/images/radio_data/augumentaion'

def fft(b, y, fs, verbose=True):
    """フィルタ係数bとフィルタされた信号yのFFTを求める"""
    b = list(b)
    y = list(y)

    N = 512    # FFTのサンプル数

    # 最低でもN点ないとFFTできないので0.0を追加
    for i in range(N):
        b.append(0.0)
        y.append(0.0)

    # フィルタ係数のFFT
    B = np.fft.fft(b[0:N])
    freqList = np.fft.fftfreq(N, d=1.0/fs)
    spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in B]

    if verbose:
        # フィルタ係数の波形領域
        subplot(221)
        plot(range(0, N), b[0:N])
        axis([0, N, -0.5, 0.5])
        xlabel("time [sample]")
        ylabel("amplitude")

        # フィルタ係数の周波数領域
        subplot(223)
        n = len(freqList) / 2
        plot(freqList[:n], spectrum[:n], linestyle='-')
        axis([0, fs/2, 0, 1.2])
        xlabel("frequency [Hz]")
        ylabel("spectrum")

        # フィルタされた波形のFFT
        Y = np.fft.fft(y[0:N])
        freqList = np.fft.fftfreq(N, d=1.0/fs)
        spectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in Y]

        # 波形を描画
        subplot(222)
        plot(range(0, N), y[0:N])
        axis([0, N, -1.0, 1.0])
        xlabel("time [sample]")
        ylabel("amplitude")

        # 振幅スペクトルを描画
        subplot(224)
        n = len(freqList) / 2
        plot(freqList[:n], spectrum[:n], linestyle='-')
        axis([0, fs/2, 0, 10])
        xlabel("frequency [Hz]")
        ylabel("spectrum")

        show()

def save(data, fs, bit, filename):
    """波形データをWAVEファイルへ出力"""
    output_path = os.path.join(OUTPUT_DIR, filename)
    wf = wave.open(output_path, "w")
    wf.setnchannels(1)
    wf.setsampwidth(bit / 8)
    wf.setframerate(fs)
    wf.writeframes(data)
    wf.close()

def filter(filepath):
    # save file name
    dir_path = os.path.dirname(filepath)
    dir_name = os.path.basename(dir_path)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    outputname = '%s_%s_fir.wav' % (dir_name, name)
    output_path = os.path.join(OUTPUT_DIR, outputname)
    if os.path.isfile(output_path):
        print("Since augumentation file already exists, the processing is skipped.")
        return outputname

    wf = wave.open(filepath, "r")
    fs = wf.getframerate()
    print("fs wf: ", fs)
    X, sample_rate = librosa.load(fn)
    fs = sample_rate
    print("fs librosa: ", fs)

    x =  wf.readframes(wf.getnframes())
    #x = wf.readframes(len(X))
    #x = frombuffer(x, dtype="int16")
    x = frombuffer(x, dtype="int16") / 32768.0

    nyq = fs / 2.0  # ナイキスト周波数

    # フィルタの設計
    # ナイキスト周波数が1になるように正規化
    fe1 = 200.0 / nyq      # カットオフ周波数1
    fe2 = 1300.0 / nyq      # カットオフ周波数2
    numtaps = 255           # フィルタ係数（タップ）の数（要奇数）

    # b = scipy.signal.firwin(numtaps, fe1)                         # Low-pass
    # b = scipy.signal.firwin(numtaps, fe2, pass_zero=False)        # High-pass
    # b = scipy.signal.firwin(numtaps, [fe1, fe2], pass_zero=False) # Band-pass
    b = scipy.signal.firwin(numtaps, [fe1, fe2])                  # Band-stop

    # FIRフィルタをかける
    y = scipy.signal.lfilter(b, 1, x)

    # フィルタ係数とフィルタされた信号のFFTを見る
    #fft(b, y, fs)

    # 音声バイナリに戻して保存
    y = [int(v * 32767.0) for v in y]
    y = struct.pack("h" * len(y), *y)

    # save file
    save(y, fs*4, 16, outputname)
    return outputname

if __name__ == '__main__':
    augumentations = []
    with open('all_labeled_data.csv', 'r') as f:
        for i, line in enumerate(f):
            try:
                file_label = line.split(",")
                fn = file_label[0]
                label = int(file_label[1])
                print("No.%d:%s" % (i, fn))
                augumentation_filename = filter(fn)
                augumentation_filepath = os.path.join(OUTPUT_DIR, augumentation_filename)
                augumentations.append((augumentation_filepath, label))
            except Exception as e:
                print("Error skip %s" % e)

    with open('all_labeld_data_aug.csv', 'w') as f:
        for augumentation in augumentations:
            line = '%s,%d' % (augumentation[0], augumentation[1])
            f.write(line)
            f.write("\n")
