import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

import DatasetUtil as datasetUtil
import NeuralNetUtil as neuralUtil

IsDataPostiveOnly = True
DATASET = pd.read_csv('datasets/FloatValueInterval[0,1](18-Decimals)Dataset.csv')


def neuralNetworkModel(dataset):
    """
    Se creeaza modelul neuronal cu arhitectura feedforward. Se apeleaza functiile necesare pentru a face transformarile
    necesare setului de date (pentru a fi compatibile pentru antrenare). Dupa crearea modelului neuronal propriu-zis,
    se vor afisa si reprezentarile grafice corespunzatoare.
    :param dataset: Setul de Date cu valori in ByteCode.
    :param isDataPositiveOnly: Daca setul de date va avea doar valori positive atunci acest bool = True, altfel False.
    :return: Crearea modelului neuronal, grafice reprezintative (prezentate mai sus).
    """

    neuralUtil.preConditions(dataset)
    dataset = neuralUtil.deserializeMainDataset(dataset)
    X, Y = neuralUtil.reformatTheDataset(dataset)
    datasetUtil.minMaxDataset(X, Y)
    datasetUtil.scatterMeanSquaredError(Y)
    X_train, X_test, Y_train, Y_test = neuralUtil.spiltDatasetIntoTrainTest(X, Y)

    print("Arhitectura Retelei Neuronale ...")

    # Model folosit din Libraia Keras, indica faptul ca se va folosi o retea neuronala.
    model = Sequential()

    # Setul de date este normalizat, coeficientii fiind in intervalul [0,1].
    # Se initializeaza greutatile cu valori intre 0.1 si 0.9.
    initializer = tf.keras.initializers.RandomUniform(minval=0.0001039519, maxval=0.9999998)

    # Se aplica conceptul de Flattening discutat si se introduce forma coeficientilor.
    # In cazul de fata, pentru sisteme de rangul 2 vor fi 6 coeficienti.
    # Dupa reformatare doua randuri si trei valori pe fiecare rand.
    model.add(Flatten(input_shape=(2, 3)))

    # Se adauga un "Dense" de 32, adica un strat ascuns cu 32 de neuroni
    # Functia de activare: Rectified Linear Unit si setarea unor hiper-paramaterii.
    # Pentru a nu face "overfitting" se regularieaza ponderile cu valori intre 10^(-7) si 10^(-4)
    model.add(Dense(32, activation='relu', kernel_initializer=initializer,
                    kernel_regularizer=regularizers.L1L2(l1=1e-9, l2=1e-9),
                    bias_regularizer=regularizers.L2(1e-9),
                    activity_regularizer=regularizers.L2(1e-9), use_bias=True))

    # Se anuleaza 10% din neuroni din strat pentru a nu face "overfit"
    model.add(Dropout(0.1))

    # Se adauga inca un strat ascuns cu 16 de neuroni
    model.add(Dense(16, activation='relu'))

    # Se anuleaza 10% din neuroni din strat pentru a nu face "overfit"
    model.add(Dropout(0.1))

    # Se adauga inca un strat ascuns cu 8 de neuroni
    model.add(Dense(8, activation='relu'))

    # Se anuleaza 20% din neuroni din strat pentru a nu face "overfit"
    model.add(Dropout(0.2))

    # Ultimul strat este de iesire si va avea 2 neuroni, fiind cele doua necunoscute
    model.add(Dense(2, activation='relu'))

    # In cazul de fata se foloseste optimizatorul Adam inloc de Stocastic gradient descent
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.0001039519,
                                    beta_2=0.9999998, epsilon=1e-09, amsgrad=True,
                                    name="Adam")

    # Functia de pierdere.
    model.compile(loss='mean_squared_logarithmic_error',
                  optimizer=adam, metrics=["accuracy", "mse", "mae"])

    # Inceperea procesului de antranare a retelei.
    history = model.fit(X_train, Y_train, epochs=50,
                        batch_size=128, verbose='auto', validation_split=0.2,
                        shuffle=True)

    print(history.history.keys())

    # Utilitara creata in
    neuralUtil.predictionAndEvaluationOfTheModel(model, X_test, Y_test)

    print(model.summary())

    neuralUtil.plot(history)
    neuralUtil.plottingModelPNG(model)
    neuralUtil.saveAndTestNeuralNetworkModel(model, X_train, Y_train, X_test)
    neuralUtil.testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(
        Y_test, False, "Keras Neural Model", False)
    neuralUtil.testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(
        model.predict(X_test), False,"Keras Neural Model", True)
    neuralUtil.sklearnNeuralNet(X_train, Y_train, X_test, Y_test,False)


if __name__ == '__main__':
    neuralNetworkModel(DATASET)
