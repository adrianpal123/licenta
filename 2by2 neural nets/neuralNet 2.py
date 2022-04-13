import os
import warnings
import DatasetUtil as datasetUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


DATASET = pd.read_csv('Datasets/CosineSimilarityDataset.csv')


def preConditions(dataset):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    print("----- PRIMELE 5 VALORI DIN SETUL DE DATE ----- " + '\n' + dataset.head())


def NormalizeDataSklearn(X,Y,revertBoolean = False):
    """
    Normalizarea valorilor sistemului intre [0,1] pentru o antrenare mai usoara.
    :param X: Coeficientii sistemului patratic [2x2].
    :param Y: Necunoscutele sistemului.
    :param revertBoolean: false daca se doreste normalizarea, true daca se doreste inversarea acestei transformari, adica
           aducerea inapoi la forma originala a valorilor.
    :return:
    """
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    if revertBoolean == False:
        X_norm = min_max_scaler.fit_transform(X)
        Y_norm = min_max_scaler.transform(Y)
        return X_norm, Y_norm
    else:
        X = min_max_scaler.inverse_transform(X)
        Y = min_max_scaler.inverse_transform(Y)
        return X,Y

def plot(history):
    """
    Se creeaza numeroase grafice pe modelul retelei neuronale creat,
    :param history: reprezinta modelul retelei neuronale artificiale.
    :return: grafic pentru accuratetea atat pentru valorile de antrenare cat si cele de testare.
             grafic pentru functia de pierdere atat pentru valorile de antrenare cat si cele de testare.
             grafic pentru eroarea patratica medie atat pentru valorile de antrenare cat si cele de testare.
    Pe aceste grafice se poate vedea foarte usor daca modelul retelei neuronale este eficace sau nu.
    """
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

if __name__ == '__main__':

    preConditions(DATASET)

    DATASET = datasetUtil.deserializeDataset(DATASET)
    print("----- AFTER THE DESERIALIZATION OF THE DATASET -----")
    for i in range(5):
        print(DATASET[i])

    sys = [datasetUtil.reformatList(list(l[0])) for l in DATASET]
    sol = [list(l[1]) for l in DATASET]
    X = datasetUtil.createX(sys, sol)
    Y = [l[2].astype("float32") for l in DATASET]
    X = np.array([np.array(val) for val in X])
    Y = np.array([np.array(val) for val in Y])

    datasetUtil.minMaxDataset(X,Y)
    datasetUtil.scatterMeanSquaredError(Y)

    """
        Se formeaza si se imparte setul de date pentru a putea fi antrenate pe modelul neuronal.
        66.(6)% din setul de date va fi pentru antrenarea propriu-zisa.
        33.(3)% din setul de date va fi pentru testarea modelului.
        dataset: setul de date din csv, in bytecode.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    print("Y[0] - Dataset", Y[0])
    print("Y[1] - Dataset", Y[1])

    # design network
    model = Sequential()
    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=9.)
    values = initializer(shape=(2, 3))
    # model.add(Convolution1D(filters=2, kernel_size=1, padding='SAME', activation='sigmoid'))
    model.add(Flatten(input_shape=(2, 3)))
    # model.add(Dense(24, activation='relu', kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02), kernel_initializer='LecunNormal', bias_initializer='zeros'))
    # model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),kernel_initializer='LecunNormal', bias_initializer='zeros' ))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='relu'))

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True, decay=1e-7, momentum=0.9)
    model.compile(loss='cosine_similarity', optimizer="Adam", metrics=["accuracy", "mse", "mae"])

    history = model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose='auto', validation_split=0.2, shuffle=True)

    print(history.history.keys())

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save("savedNeuralNetwork")

    # Se salveaza modelul neuronal
    reconstructed_model = keras.models.load_model("savedNeuralNetwork")

    # Let's check:
    np.testing.assert_allclose(model.predict(X_test), reconstructed_model.predict(X_test))

    # The reconstructed model is already compiled and has retained the optimizer
    # state, so training can resume:
    reconstructed_model.fit(X_test, Y_test)

    print(model.summary())

    plot(history)

    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

