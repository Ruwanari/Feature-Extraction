# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:07:54 2019

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
#Keras
import keras
from keras import models
from keras import layers


        

data = pd.read_csv('data.csv')
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

# spliting of dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

amaradewa = X[:4]
dest = "dest\\"
print (amaradewa)
gmm = GaussianMixture(n_components = 4, max_iter = 200, covariance_type='diag',n_init = 3)
gmm.fit(amaradewa)
picklefile = "amaradewa.gmm"
pickle.dump(gmm,open(dest + picklefile,'wb'))
print (' modeling completed for speaker: amaradewa')

'''
# creating a model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(32,activation = 'relu'))

model.add(layers.Dense(16,activation = 'relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)
                    
# calculate accuracy
test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)

# predictions
predictions = model.predict(X_test)
np.argmax(predictions[0])
'''