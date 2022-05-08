import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.regularizers import l2

import DatasetUtil as datasetUtil
import NeuralNetUtil as neuralUtil

DATASET = pd.read_csv('CurrentDatasets/CosineSimilarity[0,1]Dataset.csv')

def neuralNetworkModel(dataset):
    """
    Se creeaza modelul neuronal cu arhitectura feedforward. Se apeleaza functiile necesare pentru a face transformarile
    necesare setului de date (pentru a fi compatibile pentru antrenare). Dupa crearea modelului neuronal propriu-zis,
    se vor afisa si reprezentarile grafice corespunzatoare.
    :param dataset: Setul de Date cu valori in ByteCode.
    :return: Crearea modelului neuronal, grafice reprezintative (prezentate mai sus).
    """
    neuralUtil.preConditions(dataset)
    dataset = neuralUtil.deserializeMainDataset(dataset)
    X, Y = neuralUtil.reformatTheDataset(dataset)
    datasetUtil.minMaxDataset(X, Y)
    datasetUtil.scatterMeanSquaredError(Y)
    X_train, X_test, Y_train, Y_test = neuralUtil.spiltDatasetIntoTrainTest(X, Y)


    print("Arhitectura Retelei Neuronale ...")

    model = Sequential()
    model.add(Flatten(input_shape=(2, 3)))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
                    kernel_initializer='LecunNormal', bias_initializer='zeros'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='relu'))
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001, nesterov=True, decay=1e-7, momentum=0.8)
    model.compile(loss='cosine_similarity', optimizer=sgd, metrics=["accuracy", "mse", "mae"])
    history = model.fit(X_train, Y_train, epochs=80, batch_size=132, verbose='auto', validation_split=0.2, shuffle=True)

    print(history.history.keys())

    neuralUtil.predictionAndEvaluationOfTheModel(model, X_test, Y_test)

    print(model.summary())

    neuralUtil.plot(history)
    neuralUtil.plottingModelPNG(model)

    neuralUtil.saveAndTestNeuralNetworkModel(model,X_train,Y_test,X_test)

if __name__ == '__main__':

    neuralNetworkModel(DATASET)
