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
from keras.regularizers import l2
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


dataset = pd.read_csv('licenta_datasetTest.csv')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

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

for i in range (5):
    print(dataset[i])

sys = [reformatList(list(l[0])) for l in dataset]
sol = [list(l[1]) for l in dataset]
X = createX(sys, sol)
Y = [l[2].astype("float32") for l in dataset]

maxim = 0
for value in Y:
    if maxim < np.max(value):
        maxim = np.max(value)
print("MAXIM: ")
print(maxim)

print("First input data X:")
print(X[1])
print("First output data Y:")
print(Y[1])


X = np.array([np.array(val) for val in X])
print("Y[0] - Dataset", Y[0])
print("Y[1] - Dataset", Y[1])


X = tf.ragged.constant(X)
Y = tf.ragged.constant(Y)

print("Y[0] SHAPE - DatasetRAGGED", Y[0].shape)
print("Y[1] SHAPE - DatasetRAGGED", Y[1].shape)
X = X.to_tensor()
Y = Y.to_tensor()

print("Y[0] SHAPE - DatasetRAGGED - same length", Y[0].shape)
print("Y[1] SHAPE -  DatasetRAGGED - same length", Y[1].shape)
print(Y[0])
print(Y[1])

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

normalized_x = X
normalized_y = Y


# design network
model = Sequential()
#model.add(Convolution1D(filters=2, kernel_size=1, padding='SAME', activation='sigmoid'))
model.add(Flatten(input_shape=(9,10)))
model.add(Dense(850, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer='LecunNormal', bias_initializer='zeros'))
model.add(Dropout(0.1))
model.add(Dense(650,activation='LeakyReLU',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer='RandomUniform'))
model.add(Dropout(0.2))
model.add(Dense(500,activation='linear',kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02), kernel_initializer='LecunNormal',bias_initializer='zeros'))
model.add(Dropout(0.3))
model.add(Dense(300,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='linear', kernel_initializer='RandomUniform'))
model.add(Dropout(0.2))
model.add(Dense(150, kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02), activation='LeakyReLU'))
model.add(Dropout(0.2))
model.add(Dense(75,kernel_initializer='LecunNormal',bias_initializer='zeros', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(35,kernel_initializer='LecunNormal',bias_initializer='zeros', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(9, activation='relu'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='cosine_similarity',optimizer=sgd,metrics=["accuracy","mse","mae"])

history = model.fit(normalized_x,normalized_y , epochs=100, batch_size=128, verbose='auto',validation_split=0.2, shuffle=True)

print(history.history.keys())



print(model.summary())

plt.figure(1)

plt.subplot(211)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(212)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.tight_layout()

plt.show()


plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Mean squared error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



















