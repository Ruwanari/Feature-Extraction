# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 23:33:01 2020

@author: Ruwanari
"""

import librosa
import numpy as np
import librosa.display

audio_path1 = 'final123.wav'
audio_path2 = 'final132.wav'
audio_path3 = 'final213.wav'
audio_path4 = 'final231.wav'
audio_path5 = 'final312.wav'
audio_path6 = 'final321.wav'

y1, sr = librosa.load(audio_path1)
D1 = librosa.stft(y1)
y2, sr = librosa.load(audio_path2)
D2 = librosa.stft(y2)
y3, sr = librosa.load(audio_path3)
D3 = librosa.stft(y3)
y4, sr = librosa.load(audio_path4)
D4 = librosa.stft(y4)
y5, sr = librosa.load(audio_path5)
D5 = librosa.stft(y5)
y6, sr = librosa.load(audio_path6)
D6 = librosa.stft(y6)

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1),
                                                 ref=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('123 spectrogram')
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D2),
                                                 ref=np.max),
                        y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('132 spectrogram')
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D3),
                                                 ref=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('213 spectrogram')
plt.tight_layout()
plt.show()


plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D4),
                                                 ref=np.max),
                        y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('231 spectrogram')
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D5),
                                                 ref=np.max),
                        y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('312 spectrogram')
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D6),
                                                 ref=np.max),
                        y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('321 spectrogram')

plt.tight_layout()
plt.show()

