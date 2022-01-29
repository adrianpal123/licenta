import numpy as np
import pandas as pd
import math
from tensorflow import keras
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Convolution2D, Dropout, MaxPooling1D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import os
import tensorflow as tf



dataset = pd.read_csv('licenta_dataset3Ord.csv')

print(dataset.head())

def deserializeDataset(dataset):
  dataset_array = []
  for index, row in dataset.iterrows():
    syst = row['system'].encode()
    sol = row['solution'].encode()
    out = row['output'].encode()
    dataset_array.append([np.frombuffer(syst.decode('unicode-escape').encode('ISO-8859-1')[2:-1], dtype=np.float64),
                          np.frombuffer(sol.decode('unicode-escape').encode('ISO-8859-1')[2:-1], dtype=np.float64),
                          np.frombuffer(out.decode('unicode-escape').encode('ISO-8859-1')[2:-1], dtype=np.float64)])
  return dataset_array

dataset = deserializeDataset(dataset)


def reformatList(l):
  array_size = int(math.sqrt(len(l)))
  new_l = []
  while(l):
    new_l.append(l[:array_size])
    del(l[:array_size])
  return new_l

def createX(sys, sol):
  for i in range(len(sol)):
    for j in range(len(sol[i])):
      sys[i][j].append(sol[i][j])
      sys[i][j] = np.array(sys[i][j]).astype("float32")
  return sys



sys = [reformatList(list(l[0])) for l in dataset]
sol = [list(l[1]) for l in dataset]
X = createX(sys, sol)
Y = [l[2].astype("float32") for l in dataset]

X = X[:math.floor(9/10*len(X))]
Y = Y[:math.floor(9/10*len(Y))]
X_Test = X[math.floor(9/10*len(X)):]
Y_Test = Y[math.floor(9/10*len(Y)):]



print("-----------")
print(type(X))
#X = np.asarray(X).astype(object)
#Y = np.asarray(Y).astype(object)

print(type(X))
for i in range (len(X)):
  X[i] = np.array(X[i])

print(X[1])
print(Y[1])

#print(dataset[0])



X = np.array([np.array(val) for val in X])
#X = tf.ragged.constant(X)
#Y = tf.ragged.constant(Y)
X = np.asarray(X).astype(np.float32)
Y = np.asarray(Y).astype(np.float32)

model = Sequential()
model.add(Convolution1D(filters=256, kernel_size=1, padding='SAME', input_shape=(3,4), activation='linear'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='linear'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.02, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='mse',optimizer=sgd,metrics=['accuracy'])

model.fit(X,Y , epochs=15, batch_size=32, verbose=2)


print(model.summary())