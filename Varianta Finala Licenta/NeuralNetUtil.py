import os
import csv
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import DatasetUtil as datasetUtil

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


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
    :return: (Coeficientii dependenti / Coeficientii independenti normalizati) -> X / (Necunoscutele) - > Y
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
    :param history: Datele legate de antrenarea modelului.
    :return: Diferite grafice asupra antrentarii modelului.
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


def raggedTensorChecker(X, Y):
    """
    Daca sistemele din setul de date au rangul diferit, pentru a fi date valide pentru a intra in modelul neuronal,
    acestea trebuie transformate in ragged tensors. Altfel spus, inainte de a se apela flatten() pe datele de intrare,
    toate sistemele se vor transforma in rangul cel mai mare, iar coeficientii lipsa se vor inlocui cu 0. Trebuie sa
    intre un
    :param X: Modelul retelei neuronale creat.
    :param Y: Modelul retelei neuronale creat.
    :return: X,Y transformati in constant ragged tensors.
    """
    print("First input data X:", X[1])
    print("First output data Y:", Y[1])

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
    print("X[0] SHAPE - DatasetRAGGED - same length", X[0].shape, X[0])
    print("X[5] SHAPE -  DatasetRAGGED - same length", X[5].shape, X[5])
    print("Y[0] SHAPE - DatasetRAGGED - same length", Y[0].shape, Y[0])
    print("Y[5] SHAPE -  DatasetRAGGED - same length", Y[5].shape, Y[5])

    return X, Y


def plotMatricesLens(Y):
    """
    Se va crea un grafic cu numarul de sisteme in functie de rangul sau.
    @param Y: o matrice cu solutiile sistemelor de ecuatii.
    @return:
    """
    dictionaryArray = {}

    for row in Y:
        dictionaryArray[len(row)] = dictionaryArray.get(len(row), 0) + 1

    pd.DataFrame(dictionaryArray, index=['Rang matrice']).plot(kind='bar')

    plt.show()


def splitTrainTestForVariousRanksDataset(X, Y):
    """
    Pentru sistemele care sunt stocate in acelasi set de date, indiferent de rangul matriciilor, sistemele se vor
    imparti intr-un mod diferit in training - testing data. 80% pentru antrenare, 20% pentru testare.
    @param X: Coeficientii dependenti - independenti
    @param Y: Solutiile
    @return: Set de date pentru antrenare si testare.
    """
    X_train = X[:-20]
    X_test = X[-20:]

    Y_train = Y[:-20]
    Y_test = Y[-20:]

    return X_train, X_test, Y_train, Y_test


def predictionAndEvaluationOfTheModel(model, X_test, Y_test):
    """
    Dupa ce modelul neuronal este antrenat, se vor face predictii pe setul de date pentru testare. Se vor afisa
    adevaratele valori  dar si predictiile acestora.
    @param model: Modelul neuronal antrenat.
    @param X_test: Coeficientii dependenti si independenti (setul de date pentru testare).
    @param Y_test: Solutiile sistemelor (setul de date pentru testare).
    @return: Se afiseaza adevaratele solutii si predictiile modelului neuronal.
    """
    print("Coeficientii sistemului")
    print(X_test)

    print("Evaluarea Modelului Antrenat: ")
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("Predictia modelului ... \n--- Predictiile ---")
    print(model.predict(X_test))
    print("--- Adevaratele valori ---")
    print(Y_test)


def saveAndTestNeuralNetworkModel(model, X_train, Y_train, X_test):
    """
    Se va salva modelul retelei neuronale pentru a putea fi folosit in viitor.
    @param model:Modelul neuronal antrenat.
    @param X_train: Coeficientii dependenti si independenti 80% (setul de date pentru antrenare).
    @param Y_train: Solutiile sistemelor 80% (setul de date pentru antrenare).
    @param X_test: Coeficientii dependenti si independenti (setul de date pentru testare).
    @return: Model neuronal salvat, acesta se va putea folosi in viitor.
    """
    print("Se salveaza modelul in fisierul: saved neural network din folder-ul neural network models results")
    model.save("neural network models results/saved neural network")
    reconstructed_model = tf.keras.models.load_model("neural network models results/saved neural network")
    print("Se verifica egalitatea modelului original cu cel reconstruit.")
    np.testing.assert_allclose(model.predict(X_test), reconstructed_model.predict(X_test))
    print("Modelul reconstruit este deja compilat si a retinut optimizatorul corespunzator. Antrenamentul se va relua:")
    reconstructed_model.fit(X_train, Y_train)


def testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(testDataset, isItMultiRank, fileName, predicting):
    """
    Se vor stoca solutiile exacte si cele calculate de modelul neuronal in fisiere, pentru a putea compara rezultatele.
    @param testDataset: Solutiile sistemelor, rezultatele exacte sau cele prezise de modelul neuronal.
    @param isItMultiRank: isItMultiRank = True, daca se vor stoca, mai multe sisteme, indiferent de rang in acelasi set
    de date, isItMultiRank = False altfel.
    @param fileName: numele fisierului, diferite in functie de algoritm si daca se prezic sau nu  solutiile.
    @param predicting: boolean care specifica in care fisier trebuie scrise rezultatele. predicting = True semnifica
    faptul ca modelul va prezice solutiile si se vor stoca intr-un fisier specific, iar daca predicting = False, atunci
    se vor stoca adevaratele valori intr-un fisier (Real Solutions).
    @return: Scrierea in fisiere a solutiilor exacte si predictia solutiilor (generate de modelul neuronal.
    """
    if predicting:
        f = open('neural network models results/Predicted Solutions ' + fileName, 'w+')
    else:
        f = open('neural network models results/Real Solutions ', 'w+')

    if isItMultiRank:
        header = ['Solutie 1', 'Solutie 2', 'Solutie 3', 'Solutie 4', 'Solutie 5', '...', 'Solutie n']
    else:
        header = ['Solutie 1', 'Solutie 2']
    data = []
    writer = csv.writer(f)
    writer.writerow(header)
    for row in testDataset:
        for value in row:
            data.append(value)
        writer.writerow(data)
        data = []
    f.close()


def sklearnOutputResults(regressor, X_test, Y_test, algorithm):
    """
    Metoda ajutor pentru "sklearnModelsUtil", afiseaza informatii despre modelul neuronal cum ar fi: mae,mse,rmse.
    Se vor incerca mai multe modele neuronale (deja implementate, folosite din libraria SKLearn).
    @param regressor: modelul retelei neuronale.
    @param X_test: Coeficientii dependenti si independenti ai sistemului.
    @param Y_test: Solutiile sistemelor de ecuatii.
    @param algorithm: numele algoritmului implementat de libraria SKLearn.
    @return: Afisarea unor proprietati ale modelelor din SKLearn
    """
    print(algorithm + " SKLearn Model")
    print("Rezultatele modelului neuronal SKLearn (" + algorithm + ") au fost scrise in fisier")
    print('Mean Absolute Error' + algorithm + ' :',
          metrics.mean_absolute_error(Y_test, regressor.predict(X_test)))
    print('Mean Squared Error ' + algorithm + ' :', metrics.mean_squared_error(Y_test, regressor.predict(X_test)))
    print('Root Mean Squared Error' + algorithm + ' :',
          np.sqrt(metrics.mean_squared_error(Y_test, regressor.predict(X_test))))


def sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, algorithm, isItMultiRank):
    """
    Metoda suport pentru sklearnNeuralNet, se afiseaza date despre modelele neuronale din libraria SKLearn.
    Se vor afisa predictiile solutiilor si solutiile exacte ale sistemelor de ecuatii.
    @param regressor: modelul neuronal al retelei artificial.
    @param X_train: Coeficientii dependenti - independenti din setul de date pentru antrenare.
    @param Y_train: Solutiile din setul de date pentru antrenare.
    @param X_test: Coeficientii dependenti - independenti din setul de date pentru testare.
    @param Y_test: Solutiile din setul de date pentru testare.
    @param algorithm: numele algoritmului din libraria SKLearn
    @return: date despre un model neuronal dat din libraria SKLearn.
    """
    regressor.fit(X_train, Y_train)
    print(regressor.score(X_train, Y_train))
    print(regressor.coef_)
    print(regressor.intercept_)
    testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(regressor.predict(X_test), isItMultiRank,
                                                             "SKLearn " + algorithm + " model", True)
    testNeuralModelOutputWriteToCsvRealAndPredictedSolutions(Y_test, isItMultiRank, "SKLearn " + algorithm + " model", False)
    sklearnOutputResults(regressor, X_test, Y_test, algorithm)


def sklearnNeuralNet(X_train, Y_train, X_test, Y_test, isItMultiRank):
    """
    Aplica diferite modele neuronale din libraria SKLearn - care rezolva regressii pe setul de date.
    @param X_train: Coeficientii dependenti - independenti din setul de date pentru antrenare.
    @param Y_train: Solutiile din setul de date pentru antrenare.
    @param X_test: Coeficientii dependenti - independenti din setul de date pentru testare.
    @param Y_test: Solutiile din setul de date pentru testare.
    @return: diferite modele neuronale care rezolva regresii din libraria SKLearn.
    """
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # LinearRegression
    regressor = LinearRegression()
    sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, "LinearRegression",isItMultiRank)

    # ElasticNet
    regressor = ElasticNet()
    sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, "ElasticNet",isItMultiRank)


    """
    X_train = X_train.flatten().reshape(-1,1)
    X_test = X_test.flatten().reshape(-1,1)
    Y_train = Y_train.flatten().reshape(-1,1)
    Y_test = Y_test.flatten().reshape(-1,1)


    # SVR
    regressor = SVR()
    sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, "SVR")

    # SGDRegressor
    regressor = SGDRegressor()
    sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, "SGDRegressor")

    # KernelRidge
    regressor = KernelRidge()
    sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, "KernelRidge")

    # BayesianRidge
    regressor = BayesianRidge()
    sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, "BayesianRidge")

    # GradientBoostingRegressor
    regressor = GradientBoostingRegressor()
    sklearnModelsUtil(regressor, X_train, Y_train, X_test, Y_test, "GradientBoostingRegressor")
    """