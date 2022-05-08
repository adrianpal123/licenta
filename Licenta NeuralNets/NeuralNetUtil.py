import os
import warnings
import DatasetUtil as datasetUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def preConditions(dataset):
    """
    Verificarea folosirii tensorflow GPU, pentru o antrenare mai rapida se va folosi GPU, in locul CPU-ului.
    :param dataset: Setul de date actual
    :return: Afiseaza daca Tensorflow GPU este activat sau nu. Afiseaza primele 5 randuri din setul de date.
    """
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    print("----- PRIMELE 5 VALORI DIN SETUL DE DATE ----- ")
    print(dataset.head())


def deserializeMainDataset(dataset):
    """
    Se deserializeaza setul de date, valorile din csv fiind in bytecode -> se transforma in valori de tip float.
    :param dataset: Setul de date, valori in bytecode.
    :return: Setul de date in valori de tip float.
    """
    deserializedDataset = datasetUtil.deserializeDataset(dataset)
    print("----- AFTER THE DESERIALIZATION OF THE DATASET -----")
    for i in range(5):
        print(deserializedDataset[i])

    return deserializedDataset


def reformatTheDataset(dataset):
    """
    Pentru a intra in input layer-ul retelei neuronale, coeficientii dependenti si independeti trebuie restructurati.
    :param dataset: Setul de date care contine coeficientii dependenti, independenti si solutiile in valor de tip float.
    :return: Coeficientii dependenti si independenti refactorizati si stocati in X, iar Solutiile (valorile care trebuie
            aflate) sunt stocate in Y.
    """
    sys = [datasetUtil.reformatList(list(row[0])) for row in dataset]
    sol = [list(row[1]) for row in dataset]
    X = datasetUtil.createX(sys, sol)
    Y = [row[2].astype("float32") for row in dataset]
    X = np.array([np.array(val) for val in X])
    Y = np.array([np.array(val) for val in Y])

    return X, Y


def NormalizeDataSklearn(X, Y, revertBoolean=False):
    """
    In cazul in care in sisteme sunt prezenti coeficienti care au valori mari:
    Normalizarea valorilor sistemului intre [0,1] pentru o antrenare mai usoara.
    :param X: Coeficientii sistemului patratic [2x2].
    :param Y: Necunoscutele sistemului.
    :param revertBoolean: false daca se doreste normalizarea, true daca se doreste inversarea acestei transformari,
            adica aducerea inapoi la forma originala a valorilor.
    :return:
    """
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    if not revertBoolean:
        X_norm = min_max_scaler.fit_transform(X)
        Y_norm = min_max_scaler.transform(Y)
        return X_norm, Y_norm
    else:
        X = min_max_scaler.inverse_transform(X)
        Y = min_max_scaler.inverse_transform(Y)
        return X, Y


def spiltDatasetIntoTrainTest(X, Y):
    """
    Se formeaza si se imparte setul de date pentru a putea fi antrenate pe modelul neuronal.
    66.(6)% din setul de date va fi pentru antrenarea propriu-zisa.
    33.(3)% din setul de date va fi pentru testarea modelului.
    :param X: Coefieicentii dependenti si independenti dupa reformatarea sistemelor.
    :param Y: Solutiile sistemelor de ecuatii.
    :return: Setul de date impartit pentru o antrenare corecta.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    return X_train, X_test, Y_train, Y_test


def plot(history):
    """
    Se creeaza numeroase grafice pe modelul retelei neuronale creat,
    :param history: reprezinta modelul retelei neuronale artificiale.
    :return: grafic pentru accuratetea atat pentru valorile de antrenare cat si cele de testare.
             grafic pentru functia de pierdere atat pentru valorile de antrenare cat si cele de testare.
             grafic pentru eroarea patratica medie atat pentru valorile de antrenare cat si cele de testare.
    Pe aceste grafice se poate vedea foarte usor daca modelul retelei neuronale este eficace/eficient sau nu.
    Accuratetea ar putea fi scrisa ca numarul de predictii corecte / numarul total de predictii, fiind cea importanta,
    se poate observa cum accuratetea creste la considerabil la fiecare "Epoch" - Iteratie prin setul de date. Ajungand
    la peste 85%.
    """
    plt.figure(1)

    # plt.subplot(1)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # plt.subplot(2)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plt.tight_layout()
    plt.show()

    # plt.subplot(3)
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Mean squared error')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plottingModelPNG(model):
    """
    Se creeaza arhitectura modelului retelei neuronale de tip feedforward. Se afiseaza numarul de straturi si structura
    acestora, numarul de neuroni la fiecare strat. De asemenea, se poate observa cand se aplica flatten() si dropout().
    :param model: Modelul retelei neuronale creat.
    :return: Un PNG cu arhitectura acestui model.
    """
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

def raggedTensorChecker(X,Y):
    print("First input data X:")
    print(X[1])
    print("First output data Y:")
    print(Y[1])

    X = np.array([np.array(val) for val in X])
    Y = np.array([np.array(val) for val in Y])

    X, Y = shuffle(X, Y, random_state=0)

    print("Y[0] - Dataset", Y[0])
    print("Y[1] - Dataset", Y[1])

    X = tf.ragged.constant(X)
    Y = tf.ragged.constant(Y)

    print("Y[0] SHAPE - DatasetRAGGED", Y[0].shape)
    print("Y[1] SHAPE - DatasetRAGGED", Y[1].shape)
    X = X.to_tensor()
    Y = Y.to_tensor()

    print("Y[0] SHAPE - DatasetRAGGED - same length", Y[0].shape)
    print("Y[5] SHAPE -  DatasetRAGGED - same length", Y[5].shape)
    print(Y[0])
    print(Y[5])

    print("X[0] SHAPE - DatasetRAGGED - same length", X[0].shape)
    print("X[5] SHAPE -  DatasetRAGGED - same length", X[5].shape)
    print(X[0])
    print(X[5])

    return X,Y

def plotMatricesLens(Y):
    dictionaryArray = {}

    for row in Y:
        dictionaryArray[len(row)] = dictionaryArray.get(len(row) , 0) + 1

    names = list(dictionaryArray.keys())
    values = list(dictionaryArray.values())

    pd.DataFrame(dictionaryArray, index=['Rang matrice']).plot(kind='bar')

    plt.show()

def splitTrainTestForVariousRanksDataset(X,Y):
    X_train = X[:-20]
    X_test = X[-20:]

    Y_train = Y[:-20]
    Y_test = Y[-20:]

    return X_train, X_test, Y_train, Y_test

def predictionAndEvaluationOfTheModel(model,X_test,Y_test):
    print("Evaluarea Modelului Antrenat: ")
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("Predicting the model ... \nPREDCTIONS:")
    print(model.predict(X_test))
    print("REAL SOLUTIONS:")
    print(Y_test)

def saveAndTestNeuralNetworkModel(model,X_train,Y_train,X_test):

    print("Se salveaza modelul in fisierul: savedNeuralNetwork")
    model.save("savedNeuralNetwork")
    reconstructed_model = tf.keras.models.load_model("savedNeuralNetwork")
    print("Se verifica modelul reconstruit.")
    np.testing.assert_allclose(model.predict(X_test), reconstructed_model.predict(X_test))
    print("Modelul reconstruit este deja compilat si a retinut optimizatorul corespunzator. Antrenamentul se va relua:")
    reconstructed_model.fit(X_train, Y_train)