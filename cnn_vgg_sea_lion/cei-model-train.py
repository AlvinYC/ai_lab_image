
#%matplotlib inline
import time
import numpy as np
#import cv2
#import matplotlib.pyplot as plt
import skimage.feature
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tempfile import TemporaryFile

#img_path = "../corpus/KaggleNOAASeaLions"
#img_path = "../corpus/KaggleNOAASeaLions_small"
#img_path = "../corpus/KaggleNOAASeaLions_mini"




r = 0.4     #scale down
width = 100 #patch size

final_trainX = np.load('o_trainx.npy')
final_trainY = np.load('o_trainy.npy')


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(width,width,3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='linear'))


start_time = time.time()
optim = keras.optimizers.SGD(lr=1e-5, momentum=0.2)
model.compile(loss='mean_squared_error', optimizer=optim)
model.fit(final_trainX, final_trainY, epochs=8, verbose=2)
end_time = time.time()
print('\nspend ' + str(end_time-start_time)+'s')

start_time = time.time()
optim = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=optim)
model.fit(final_trainX, final_trainY, epochs=30, verbose=2)
end_time = time.time()
print('\nspend ' + str(end_time-start_time)+'s')

model.save('cei-sea-lion-model.h5')

