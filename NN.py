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


for i in range (5):
    print(dataset[i])

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
print("INPUT IS: ")
print(X.shape)


# design network
model = Sequential()
model.add(Dense(10, input_dim=(3,4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
model.fit(X, Y, epochs=100, batch_size=len(X), verbose=0)

''''
model = Sequential([
    Dense(32, activation='relu', input_shape=(3,4)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

##### Step 4 - Compile keras model
model.compile(optimizer='sgd', # default='rmsprop', an algorithm to be used in backpropagation
              loss='binary_crossentropy', # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
              metrics=['Accuracy', 'Precision', 'Recall'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
              loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
              weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
              run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
              steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
             )


##### Step 5 - Fit keras model on the dataset
model.fit(X, # input data
          Y, # target data
          batch_size=10, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=3, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
          verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
          callbacks=None, # default=None, list of callbacks to apply during training. See tf.keras.callbacks
          validation_split=0.2, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
          #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
          shuffle=True, # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
          class_weight={0 : 0.3, 1 : 0.7}, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
          sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
          initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
          steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
          validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
          validation_batch_size=None, # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
          validation_freq=3, # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
          max_queue_size=10, # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
          workers=1, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
          use_multiprocessing=False, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
         )

'''''