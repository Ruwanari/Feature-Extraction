# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:24:46 2019

@author: Ruwanari
"""

import noisereduce as nr
import librosa
# load data

rate, data = librosa.read("newfile")
# select section of data that is noise
noisy_part = data[10000:15000]
# perform noise reduction
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)