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


        

data = pd.read_csv('data_spleeter.csv')
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
'''
amaradewa = X[:4]
chandralekha = X[4:8]
greshan = X[8:12]
nanda = X[12:16]
'''
amal = X[0:12]
amaradewa = X[12:22]
amarasiri = X[22:31]
anjaline = data.iloc[32:41]
chandraleka = X[41:49]
clarence = X[49:59]
dayan = X[59:69]
deepika = X[71:81]
diwulgane = X[79:89]
greshan = X[88:98]
indrani = X[99:109]
jagath = X[109:119]
jothipala = X[119:129]
kapuge = X[129:139]
kasun = X[139:149]
latha = X[149:159]
malani = X[159:169]
mervin = X[169:179]
milton = X[179:189]
nanda = X[189:199]
neela = X[199:209]
nelu = X[209:219]
nirosha = X[219:229]
ranil = X[229:239]
rookantha = X[239:249]
samitha = X[249:259]
shashika = X[259:269]
sujatha = X[269:279]
tm = X[279:287]
umaria = X[289:298]
'''
artists = [amal, amaradewa, amarasiri, anjaline, chandraleka, clarence, dayan, deepika, diwulgane, greshan, indrani, jagath, jothipala, kapuge, kasun, latha, malani, mervin, milton, nanda, neela, nelu, nirosha, ranil, rookantha, samitha, shashika, sujatha, tm, umaria]
# spliting of dataset into train and test dataset
'''
artists = [amal,greshan]
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''
dest = "destSpleeter3\\"
t = 0
for artist in artists:
    gmm = GaussianMixture(n_components = 4, max_iter = 200, covariance_type='diag',n_init = 3)
    gmm.fit(artist)
    picklefile = str(t)+".gmm"
    pickle.dump(gmm,open(dest + picklefile,'wb'))
    print (' modeling completed for speaker:'+str(t))
    t = t+1

'''
# creating a model
model = models.Sequential()
model.add(layers.Dense(240, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(120, activation='relu'))

model.add(layers.Dense(60, activation='relu'))

model.add(layers.Dense(30,activation = 'softmax'))
'''
'''
model.add(layers.Dense(16,activation = 'relu'))

model.add(layers.Dense(10, activation='softmax'))
'''
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=20)
                    
# calculate accuracy
test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)

# predictions
predictions = model.predict(X_test)
np.argmax(predictions[0])

'''