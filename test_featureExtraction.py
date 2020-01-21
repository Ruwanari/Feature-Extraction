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
import matplotlib.pyplot as plt

#creating headers
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()
file = open('data_test_harrep.csv', 'w', newline='',encoding='utf-8')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
artists = 'amal amaradewa amarasiri anjaline chandralekha clarence dayan deepika diwulgane greshan indrani jagath jothipala kapuge kasun latha malani mervin milton nanda neela nelu nirosha ranil rookantha samitha shashika sujatha tm umaria'.split()
for g in artists:
    for filename in os.listdir(f'./test_dataset/{g}'):
        songname = f'./test_dataset/{g}/{filename}'
        y, sr = librosa.load(songname)
        D = librosa.stft(y)
        H, P = librosa.decompose.hpss(D)
        harmonic = librosa.istft(H)
        librosa.output.write_wav('har.wav',harmonic,sr)
        z, sr = librosa.load('har.wav')
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
        file = open('data_test_harrep.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
data = pd.read_csv('data_test_harrep.csv')
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