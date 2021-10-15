# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import glob
import math


def fft_mag(x):
    # Analysis of a signal using the discrete Fourier transform
    # x: input signal, w: analysis window, N: FFT size
    # returns spectrum of the block
    # Size of positive side of fft
    blockSize = len(x)
    # Define window
    w = np.hanning(blockSize)
    w = w / np.sum(w)
    x = x * w
    relevant = (blockSize // 2) + 1
    h1 = (blockSize + 1) // 2
    h2 = blockSize // 2
    # Arrange audio to center the fft around zero
    x_arranged = np.zeros(blockSize)
    x_arranged[:h1] = x[h2:]
    x_arranged[-h2:] = x[:h2]
    # compute fft and keep the relevant part
    X = np.fft.fft(x_arranged)[:relevant]
    # compute magnitude spectrum in dB
    magX = abs(X)

    return magX


def compute_stft(xb):
    # Generate spectrogram
    # returns magnitude spectrogram

    blockSize = xb.shape[1]
    hopSize = int(blockSize / 2)

    mag_spectrogram = []
    for block in xb:
        magX = fft_mag(block)
        mag_spectrogram.append(np.array(magX))

    mag_spectrogram = np.array(mag_spectrogram)
    return mag_spectrogram


def extract_rms(xb):
    # Returns an array (NumOfBlocks X k) of spectral flux for all the audio blocks: k = frequency bins
    # xb is a matrix of blocked audio data (dimension NumOfBlocks X blockSize)

    rms_dB = []
    for block in xb:
        rms = np.sqrt(np.dot(block, block)/len(block))
        # Replace rms<-100dB with -100dB. -100dB = 10^(-5)
        if rms <= 10**(-5):
            rms = 10**(-5)
        rms = 20*np.log10(rms)
        rms_dB.append(rms)
    rms_dB = np.array(rms_dB)
    return rms_dB


def extract_spectral_crest(xb):
    # Calculate STFT
    X = compute_stft(xb)

    spectral_crest = np.array([])
    for frame in X:
        crest_factor = np.max(abs(frame)) / np.sum(abs(frame))
        spectral_crest = np.append(spectral_crest, crest_factor)

    return spectral_crest


def extract_spectral_flux(xb):
    # Returns an array (NumOfBlocks X k) of spectral flux for all the audio blocks: k = frequency bins
    # xb is a matrix of blocked audio data (dimension NumOfBlocks X blockSize)

    X = compute_stft(xb)

    # Compute spectral flux
    # Initialise blockNum and freqIndex
    n = 0
    k = 0

    spectral_flux = np.zeros(xb.shape[0])

    for n in np.arange(X.shape[0] - 1):
        flux_frame = 0
        for k in np.arange(X.shape[1]):
            flux = (abs(X[n + 1, k]) - abs(X[n, k])) ** 2
            flux_frame += flux
        flux_frame = np.sqrt(flux_frame) / (xb.shape[1] // 2 + 1)
        spectral_flux[n] = flux_frame
    spectral_flux = np.array(spectral_flux)

    return spectral_flux


def extract_spectral_centroid(xb, fs):
    #xb_afterFFT = np.zeros((np.shape(xb)[0], np.shape(xb)[1]))
    #for flag, i in enumerate(xb):
    #    iw = i * np.hanning(np.shape(xb)[1])
    #    iw = scipy.fftpack.fft(iw)
    #    xb_afterFFT[flag] = iw
    xb_afterFFT = compute_stft(xb)
    vsc = np.zeros(np.shape(xb_afterFFT)[0])
    half_k = int(np.shape(xb_afterFFT)[1]/2)
    for flag_centroid, block in enumerate(xb_afterFFT):
        left = np.arange(half_k)
        vsc[flag_centroid] = np.sum(left * xb_afterFFT[flag_centroid, 0:half_k]) / np.sum(xb_afterFFT[flag_centroid, 0:half_k])
    vsc = (vsc/1024) * fs
    return vsc


def extract_zerocrossingrate(xb):
    vzc = np.zeros(np.shape(xb)[0])
    for n, block in enumerate(xb):
        vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign(block))))
    return vzc


#def block_audio(x, blockSize, hopSize, fs):
#    blockNum = int(len(x) / hopSize)
#    add0 = int(blockSize - len(x) % hopSize)
#    x = np.hstack((x, np.zeros(add0)))
#    block = np.zeros((blockNum, blockSize))
#    flag = 0
#    for i in np.arange(blockNum):
#        block[i] = x[flag:flag + blockSize]
#        flag = flag + hopSize
#    timeInSec = np.arange(0, flag - hopSize, blockSize / fs)
#    return block, timeInSec


def block_audio(x, blockSize, hopSize, fs):
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t


def extract_features(x, blockSize, hopSize, fs):
    block, timeInSec = block_audio(x, blockSize, hopSize, fs)

    Vsc = extract_spectral_centroid(block, fs)
    Vzc = extract_zerocrossingrate(block)
    features = np.vstack((Vsc, Vzc))

    rms = extract_rms(block)
    features = np.vstack((features, rms))
    spectral_flux = np.hstack((extract_spectral_flux(block), [0]))
    features = np.vstack((features, spectral_flux))
    spectral_crest = extract_spectral_crest(block)
    features = np.vstack((features, spectral_crest))

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


def visualize_features(path_to_musicspeech):
    ndarray1 = np.arange(10)
    ndarray2 = np.arange(10)
    data = (ndarray1, ndarray2)
    plt.figure()
    plt.subplot(3, 2, 1)
    y1 = data[:, 0]
    y2 = data[:, 1]
    plt.scatter(np.shape(ndarray1)[1], y1, color='red')
    plt.scatter(np.shape(ndarray1)[1], y2, color='blue')


def normalize_zscore(featureData):
    return featureData


m = get_feature_data("D:\SchoolWork\ACA\ACAassign2\music_speech\music_wav", 1024, 256)
m = normalize_zscore(m)


