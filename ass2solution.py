# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import glob
import math

def extract_spectral_centroid(xb, fs):
    xb_afterFFT = np.zeros((np.shape(xb)[0], np.shape(xb)[1]))
    for flag, i in enumerate(xb):
        iw = i * np.hanning(np.shape(xb)[1])
        iw = scipy.fftpack.fft(iw)
        xb_afterFFT[flag] = iw
    xb_afterFFT = np.absolute(xb_afterFFT)
    vsc = np.zeros(np.shape(xb_afterFFT)[0])
    half_k = np.shape(xb_afterFFT)[1]/2
    for flag_centroid, block in enumerate(xb_afterFFT):
        left = math.log(np.arange(np.shape(xb_afterFFT)[1]/2)*flag_centroid/fs, 2)
        vsc[flag_centroid] = np.sum(left * xb_afterFFT[flag_centroid, 0:half_k]) / np.sum(xb_afterFFT[flag_centroid, 0:half_k])
    return vsc


def extract_zerocrossingrate(xb):
    vzc = np.zeros(np.shape(xb)[0])
    for n, block in enumerate(xb):
        vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign[block])))
    return vzc


def block_audio(x, blockSize, hopSize, fs):
    blockNum = int(len(x) / hopSize)
    add0 = int(blockSize - len(x) % hopSize)
    x = np.hstack((x, np.zeros(add0)))
    block = np.zeros((blockNum, blockSize))
    flag = 0
    for i in np.arange(blockNum):
        block[i] = x[flag:flag + blockSize]
        flag = flag + hopSize
    timeInSec = np.arange(0, flag - hopSize, blockSize / fs)
    return block, timeInSec


def extract_features(x, blockSize, hopSize, fs):
    block, timeInSec = block_audio(x, blockSize, hopSize, fs)
    Vsc = extract_spectral_centroid(block, fs)
    Vzc = extract_zerocrossingrate(block)
    features = np.vstack(Vsc, Vzc)
    return features


def aggregate_feature_per_file(features):
    feature_matrix = np.zeros(10)
    for n, element in enumerate(features):
        feature_matrix[n] = np.mean(element)
        feature_matrix[n+5] = np.std(element)
    return feature_matrix


def get_feature_data(path, blockSize, hopSize):
    all_feature_matrix = np.zeros((0, 10))
    folder = glob.glob(path + "\\*.wav")
    for count, wavFile in enumerate(folder):
        fs, x = scipy.io.wavfile.read(wavFile)
        all_feature_matrix = np.vstack((all_feature_matrix, aggregate_feature_per_file(extract_features(x, blockSize, hopSize, fs))))
    return all_feature_matrix
