# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:24:27 2020

@author: Ruwanari
"""

import os
import pickle
import numpy as np
from scipy.io.wavfile import read
import warnings
warnings.filterwarnings("ignore")
import time

from sklearn import preprocessing
import python_speech_features as mfcc
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined
 
#path to training data
source   = "spleeterDataset"   
modelpath = "speakersModels2\\"
test_file = "test.txt"
file_paths = open(test_file,'r')
 
gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
 
#Load the Gaussian gender Models
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname
              in gmm_files]
 
# Read the test directory and get the list of test audio files
for path in file_paths:   
 
    path = path.strip()
    print (path)
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
 
    log_likelihood = np.zeros(len(models)) 
 
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
 
    winner = np.argmax(log_likelihood)
    print ("\tdetected as - ", speakers[winner])
    time.sleep(1.0)