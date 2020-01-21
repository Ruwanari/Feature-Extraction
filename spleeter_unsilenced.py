# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:41:52 2020

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

from pydub import AudioSegment


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

#creating headers
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()
file = open('data_spleeter_unsilenced.csv', 'w', newline='',encoding='utf-8')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
artists = 'amal amaradewa amarasiri anjaline chandralekha clarence dayan deepika diwulgane greshan indrani jagath jothipala kapuge kasun latha malani nanda neela nelu'.split()
for g in artists:
    for filename in os.listdir(f'./spleeterDataset/{g}'):
        songname = f'./spleeterDataset/{g}/{filename}'
        sound = AudioSegment.from_file(songname, format="wav")

        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())
        duration = len(sound)    
        trimmed_sound = sound[start_trim:duration-end_trim]
        trimmed_sound.export("trimmed.mp3",format = "mp3")
        sname = "trimmed.mp3"
        z, sr = librosa.load(sname)
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
        file = open('data_spleeter_unsilenced.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
data = pd.read_csv('data_spleeter_unsilenced.csv')
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