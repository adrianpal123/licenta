import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

import DatasetUtil as datasetUtil
import NeuralNetUtil as neuralUtil

IsDataPostiveOnly = False
mylist = []

for chunk in pd.read_csv('datasets/Systsems10csv', chunksize=1000):
    mylist.append(chunk)

DATASET = pd.concat(mylist, axis=0)

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
    initializer = tf.keras.initializers.RandomUniform(minval=-0.26519078, maxval=0.9)
    model.add(Flatten(input_shape=(10, 11)))
    model.add(Dense(120, activation=activationFunction, kernel_initializer=initializer, use_bias=True))
    model.add(Dropout(0.1))
    model.add(Dense(60, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=activationFunction))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=-0.26519078, beta_2=0.9, epsilon=1e-09, amsgrad=True,
                                    name="Adam")
    model.compile(loss='mse', optimizer=adam, metrics=["accuracy", "mse", "mae"])
    history = model.fit(X_train, Y_train, epochs=20, batch_size=64, verbose='auto', validation_split=0.3,
                        shuffle=True)

    print(history.history.keys())

    neuralUtil.predictionAndEvaluationOfTheModel(model, X_test, Y_test)

    print(model.summary())

    neuralUtil.plot(history)
    neuralUtil.plottingModelPNG(model)
    neuralUtil.saveAndTestNeuralNetworkModel(model, X_train, Y_train, X_test)
    neuralUtil.testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(Y_test, True, "Keras Neural Model", False)
    neuralUtil.testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(model.predict(X_test), True,
                                                                        "Keras Neural Model", True)
    neuralUtil.sklearnNeuralNet(X_train, Y_train, X_test, Y_test, True)
    neuralUtil.timeDifferences(X_test, 10)


if __name__ == '__main__':
    neuralNetworkModel(DATASET, IsDataPostiveOnly)
