# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:57:59 2019

@author: Ruwanari
"""
import pickle
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time


data = pd.read_csv('data_test_Spleeter1.csv')
data.head()
# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
data.head()
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# normalizing
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
amal = X[0:1]
deepika = X[1:2]
amarasiri = X[2:3]
modelpath = "destSpleeter3\\"
gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]


log_likelihood = np.zeros(len(models))
for i in range(len(models)):
    gmm    = models[i]  #checking with each model one by one
    scores = np.array(gmm.score(amal))
    log_likelihood[i] = scores.sum()
     
winner = np.argmax(log_likelihood)
print ("\tdetected as - ", speakers[winner])
time.sleep(1.0)
print (log_likelihood)