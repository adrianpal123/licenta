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
import os
from sklearn import preprocessing
from keras.regularizers import l2
import matplotlib.pyplot as plt



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
dataset = pd.read_csv('licenta_dataset3.csv')

print("DATASET HEAD:")
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

print(len(dataset))

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



X = np.array([np.array(val) for val in X])
print(X.shape)

X = np.asarray(X).astype(np.float32)
Y = np.asarray(Y).astype(np.float32)

maxim = 0
for value in Y:
    if maxim < np.max(value):
        maxim = np.max(value)
print("MAXIM: ")
print(maxim)

print(X.shape)

print("X SHAPE IS")
print(X[1].shape)
print("Y SHAPE IS")
print(Y[1].shape)
print(X[1])
print(Y[1])


#Normalize Data Between 0 And 1
def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

normalized_x = X #NormalizeData(X)
normalized_y = Y #NormalizeData(Y)



print(normalized_x[1])
print(normalized_y[1])

model = Sequential()
#model.add(Convolution2D(filters=256, kernel_size=1, padding='SAME', input_shape=(3,4), activation='sigmoid'))
model.add(Flatten(input_shape=(3,4)))
model.add(Dense(715, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(Dense(515,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(215,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),kernel_initializer = tf.keras.initializers.Identity(),bias_initializer='zeros', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(75, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='relu'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='cosine_similarity',optimizer=sgd,metrics=["accuracy"])

history = model.fit(normalized_x,normalized_y , epochs=100, batch_size=320, verbose='auto', validation_split=0.2,shuffle=True)


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
