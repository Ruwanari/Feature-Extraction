# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:00:15 2019

@author: Ruwanari
"""

from __future__ import print_function
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import csv
# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
#Keras
import librosa.display

#creating headers
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()
file = open('data_test.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for filename in os.listdir(f'./test_dataset'):
    songname = f'./test_dataset/{filename}'
    y, sr = librosa.load(songname)
    S_full, phase = librosa.magphase(librosa.stft(y))
    idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
    S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    D_foreground = S_foreground * phase
    y_foreground = librosa.istft(D_foreground)
    librosa.output.write_wav('newfile', y_foreground, sr)
    newfile = 'newfile'
    z,sr = librosa.load(newfile, mono=True, duration=240)
    chroma_stft = librosa.feature.chroma_stft(y=z, sr=sr)
    rmse = librosa.feature.rmse(y=z)
    spec_cent = librosa.feature.spectral_centroid(y=z, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=z, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=z, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(z)
    mfcc = librosa.feature.mfcc(y=z, sr=sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
            to_append += f' {np.mean(e)}'
    to_append += f' {filename}'
    file = open('data_test.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
data = pd.read_csv('data_test.csv')
data.head()
# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
data.head()
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))