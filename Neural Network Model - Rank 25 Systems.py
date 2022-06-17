import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Flatten, Dropout

import DatasetUtil as datasetUtil
import NeuralNetUtil as neuralUtil

IsDataPostiveOnly = False
mylist = []

for chunk in  pd.read_csv('datasets/SystemSize25(18-Decimal)Dataset.csv', chunksize=1000):
    mylist.append(chunk)

DATASET = pd.concat(mylist, axis= 0)

del mylist



def neuralNetworkModel(dataset, isDataPositiveOnly):
    """
    Se creeaza modelul neuronal cu arhitectura feedforward. Se apeleaza functiile necesare pentru a face transformarile
    necesare setului de date (pentru a fi compatibile pentru antrenare). Dupa crearea modelului neuronal propriu-zis,
    se vor afisa si reprezentarile grafice corespunzatoare.
    :param dataset: Setul de Date cu valori in ByteCode.
    :param isDataPositiveOnly: Daca setul de date va avea doar valori positive atunci acest bool = True, altfel False.
    :return: Crearea modelului neuronal, grafice reprezintative (prezentate mai sus).
    """
    if isDataPositiveOnly:
        print("Coeficientii si solutiile sistemului sunt doar valori positive.")
        activationFunction = "relu"
    else:
        print("Coeficientii si solutiile sistemului NU sunt doar valori positive.")
        activationFunction = "LeakyReLU"
    neuralUtil.preConditions(dataset)
    dataset = neuralUtil.deserializeMainDataset(dataset)
    X, Y = neuralUtil.reformatTheDataset(dataset)
    datasetUtil.minMaxDataset(X, Y)
    X_train, X_test, Y_train, Y_test = neuralUtil.spiltDatasetIntoTrainTest(X, Y)

    print(type(X_train))

    print("Arhitectura Retelei Neuronale ...")

    model = Sequential()
    initializer = tf.keras.initializers.RandomUniform(minval=0.1, maxval=0.9)
    model.add(Flatten(input_shape=(25, 26)))
    model.add(Dense(640, activation=activationFunction, kernel_initializer=initializer,use_bias=True))
    model.add(Dropout(0.1))
    model.add(Dense(480, activation=activationFunction))
    model.add(Dropout(0.1))
    model.add(Dense(300, activation=activationFunction))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation=activationFunction))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01,beta_1=0.1,beta_2=0.9,epsilon=1e-09,amsgrad=True,name="Adam")
    model.compile(loss='mse', optimizer=adam, metrics=["accuracy", "mse", "mae"])
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose='auto', validation_split=0.2,
                        shuffle=True)

    print(history.history.keys())

    neuralUtil.predictionAndEvaluationOfTheModel(model, X_test, Y_test)

    print(model.summary())

    neuralUtil.plot(history)
    neuralUtil.plottingModelPNG(model)
    neuralUtil.saveAndTestNeuralNetworkModel(model, X_train, Y_train, X_test)
    neuralUtil.testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(Y_test, False,"Keras Neural Model",False)
    neuralUtil.testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(model.predict(X_test), False, "Keras Neural Model",True)
    neuralUtil.sklearnNeuralNet(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    neuralNetworkModel(DATASET, IsDataPostiveOnly)
