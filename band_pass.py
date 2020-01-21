# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:07:46 2019

@author: Ruwanari
"""

from scipy.io import wavfile
from scipy import signal
import numpy as np
import librosa

#audio_path = 'Buddanubawena.mp3'
#x , sr = librosa.load(audio_path)
#   sr, x = wavfile.read('Buddanubawena.mp3')      # 16-bit mono 44.1 khz



#   b = signal.firwin(101, [0.1, 0.9], pass_zero=False)

#   x = signal.lfilter(b, [1.0], x)


#y = x.astype(np.int16)
#librosa.output.write_wav('bandpasstest1', x.astype(np.int16), sr)
#   wavfile.write('bandpasstest1.wav', sr, x.astype(np.int16))

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math
import contextlib
from scipy.signal import butter, lfilter


fname = 'Buddanubawena.wav'
outname = 'bp6.wav'

lo,hi=85,2400
y,sr = librosa.load(fname)
b,a=butter(N=6, Wn=[2*lo/sr, 2*hi/sr], btype='band')
x = lfilter(b,a,y)
librosa.output.write_wav(outname, x, sr)
# sounddevice.play(x, sr)  # playback

'''
cutOffFrequency = 400.0

# from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

# from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

with contextlib.closing(wave.open(fname,'rb')) as spf:
    sampleRate = spf.getframerate()
    ampWidth = spf.getsampwidth()
    nChannels = spf.getnchannels()
    nFrames = spf.getnframes()

    # Extract Raw Audio from multi-channel Wav File
    signal = spf.readframes(nFrames*nChannels)
    spf.close()
    channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

    # get window size
    # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
    freqRatio = (cutOffFrequency/sampleRate)
    N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

    # Use moviung average (only on first channel)
    filtered = running_mean(channels[0], N).astype(channels.dtype)

    wav_file = wave.open(outname, "w")
    wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
    wav_file.writeframes(filtered.tobytes('C'))
    wav_file.close()




from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    z, sr = librosa.core.load(fname)
    plt.figure(2)
    plt.clf()
    plt.plot(t, z, label='Noisy signal')

    y = butter_bandpass_filter(z, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()


import wave
import numpy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import filtfilt

# open the audio file and extract some information
spf = wave.open('Buddanubawena.wav','r')
(nChannels, sampWidth, sampleRate, nFrames, compType, compName) = spf.getparams()

# extract audio from wav file
input_signal = spf.readframes(-1)
input_signal = numpy.fromstring(input_signal, 'Int16')
spf.close()

amp = 1.0
input_signal = amp * input_signal / max(abs(max(input_signal)),abs(min(input_signal)))
# create the filter
N = 4
nyq = 0.5 * sampleRate
low = 100 / nyq
high = 500 / nyq
b, a = signal.butter(N, [low, high], btype='band')

# apply filter
output_signal = signal.filtfilt(b, a, input_signal)

# ceate output file
wav_out = wave.open("output.wav", "w")
wav_out.setparams((nChannels, sampWidth, sampleRate, nFrames, compType, compName))

# write to output file
wav_out.writeframes(output_signal.tobytes())
wav_out.close()

# plot the signals
t = numpy.linspace(0, nFrames/sampWidth, nFrames, endpoint = False)
plt.plot(t, input_signal, label='Input')
plt.plot(t, output_signal, label='Output')
plt.show()
'''