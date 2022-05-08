import numpy as np
import pandas as pd
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Pentru sisteme de rang 2 -> 10^8*6 cazuri diferite -> 10 cifre, 8 decimale, 6 valori diferite.
# un array cu diverse datasets de sisteme patratice deterministe de rang 2.
datasets = ['Datasets/licenta_dataset2maxMin50.csv', 'Datasets/CosineSimilarityDataset.csv',
            'Datasets/licenta_dataset2maxMin50.csv', 'Datasets/NoLimitDatasetNegativePositive.csv']


def deserializeDataset(dataset):
    """
    Transforma bytecode-ul in text.
    :param dataset: reprezinta fisierul de tip csv (baza de date) unde sunt stocate valorile sistemelor de ecuatii.
    :return: valorile deseralizate, 3 array-uri -> coeficientii dependenti / coeficientii independenti / necunoscutele.
    """
    print("Numarul de sisteme de ecuatii in setul de date: " + '\n' + str(len(dataset)))
    dataset_array = []
    for index, row in dataset.iterrows():
        syst = row['system'].encode()
        sol = row['solution'].encode()
        out = row['output'].encode()
        dataset_array.append([np.frombuffer(syst.decode('unicode-escape').encode('ISO-8859-1')[2:-1], dtype=np.float64),
                              np.frombuffer(sol.decode('unicode-escape').encode('ISO-8859-1')[2:-1], dtype=np.float64),
                              np.frombuffer(out.decode('unicode-escape').encode('ISO-8859-1')[2:-1], dtype=np.float64)])
    return dataset_array


def reformatList(l):
    """
    Reformateaza array-ul primit, in asa fel incat sa fie sub forma unei matrice.
    :param l: un array de lungime 4.
    :return: o matrice [2x2].
    """
    array_size = int(math.sqrt(len(l)))
    new_l = []
    while (l):
        new_l.append(l[:array_size])
        del (l[:array_size])
    return new_l


def createX(sys, sol):
    """
    La fiecare linie i se alatura si coeficientul independent (numarul din partea drepta a semnului egal).
    EX:  |  1*X + 1*Y = 2
         | -5*X + 3*Y = 0,
         Unde X = 1.25, Y = 0.75.
    Se vor muta coeficientii independenti in partea stanga, valorile care vor intra in reteau neuronala fiind urmatoarele:
    test_input = [[1,1,2],[-5,3,0]]
    test_target = [1.25,0.75]
    :param sys: coeficientii dependenti.
    :param sol: coeficientii independenti.
    :return: o matrice cu coeficientii independenti lipiti de coeficientii dependenti.
    """
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            sys[i][j].append(sol[i][j])
            sys[i][j] = np.array(sys[i][j]).astype("float32")
    return sys


def minMaxDataset(X,Y):
    """
    Se cauta valorile minime si maxime dintre coeficienti si necunoscute.
    :param X: Coeficientii sistemului patratic [2x2].
    :param Y: Necunoscutele sistemului.
    :return: Se afiseaza varile maxime si minime dintre coeficientii dependenti, independenti si valori.
    """
    maxim = 0
    for value in Y:
        if maxim < np.max(value):
            maxim = np.max(value)
    print("---------VALOARE MAXIMA -> NECUNOSCUTA: " + str(maxim))

    minim = 999999999999
    for value in Y:
        if minim > np.max(value):
            minim = np.min(value)
    print("---------VALOARE MINIMA -> NECUNOSCUTA: " + str(minim))

    maxim = 0
    for value in X:
        if maxim < np.max(value):
            maxim = np.max(value)
    print("---------VALOARE MAXIMA -> COEFICIENT: " + str(maxim))

    minim = 999999999999
    for value in X:
        if minim > np.max(value):
            minim = np.min(value)
    print("---------VALOARE MINIMA -> COEFICIENT: " + str(minim))


def plotMatricesLens(Y):
    """
    Se pune in grafic numarul total de sisteme din dataset-ul specificat.
    :param Y: Un array cu necunoscutele fiecarui sistem.
    :return: numarul de sisteme prezente in dataset pus in grafic.
    """
    dictionaryArray = {}

    for row in Y:
        dictionaryArray[len(row)] = dictionaryArray.get(len(row), 0) + 1

    pd.DataFrame(dictionaryArray, index=['Rang matrice']).plot(kind='bar')

    plt.show()


def scatterPoints(Y):
    """
    Grafic cu necunoscutele sistemlor de ecuatii.
    :param Y: Un array cu necunoscutele fiecarui sistem
    :return:
    """
    y_1, y_2 = Y.T
    plt.scatter(y_1, y_2)
    plt.title("solutiilor sistemelor de tip A(X,Y), A_1(X,Y), ..., A_n(X,Y)")
    plt.show()


def scatterMeanSquaredError(Y):
    """
    Se traseaza linia polyFit, reprezinta linia cea mai aproape de toate punctele din plan sau zis altfel
    suma dinstantelor dintre puncte si linie sa fie minima.
    :param Y:
    :return:
    """
    y_1, y_2 = Y.T
    plt.scatter(y_1, y_2)

    m, b = np.polyfit(y_1, y_2, 1)

    plt.plot(y_1, m * y_1 + b, color='C2')
    plt.title("Linia PolyFit - distanta dintre puncte si linie sa fie minima")
    plt.show()


def plotSystemof2ndRang(a, b, c):
    """
    Se reprezinta in grafic primul sistem din setul de date.
    :param a: Coeficientii Dependenti ai sistemului patratic.
    :param b: Coeficientii Independenti ai sistemului patratic.
    :param c: Necunoscutele sistemului patratic.
    :return: Reprezentatia grafica a sistemului de ecuatii liniare.
    """
    print("Coeficientii Dependenti")
    print(a)
    print("Coeficientii Independenti")
    print(b)
    print("Solutiile Sistemului")
    print(c)

    plt.figure()

    # Set x-axis range
    plt.xlim((-10000, 10000))
    # Set y-axis range
    plt.ylim((-10000, 10000))
    # Draw lines to split quadrants
    plt.plot([-10000, 10000], [0, 0], color='C0')
    plt.plot([0, 0], [-10000, 10000], color='C0')

    x = np.linspace(-10000, 10000)

    # a[0]x + a[1] y = b => y = (b - a[0]x)/a[1]
    y = ((-a[0]) * x + b[0]) / a[1]
    plt.plot(x, y, color='C2')

    # a[2]*x + a[3]*y = b => y = (b - a[2]x)/a[3]
    y = ((-a[2]) * x - b[1]) / a[3]
    plt.plot(x, y, color='C2')

    # Add solution
    plt.scatter(c[0], c[1], marker='x', color='black')
    # Annotate solution
    plt.annotate('({:0.3f}, {:0.3f})'.format(c[0], c[1]), c + 0.5)

    plt.title('Primul sistem din dataset')

    plt.show()


if __name__ == '__main__':

    for value in datasets:
        dataset = pd.read_csv(value)
        print(dataset.head())
        dataset = deserializeDataset(dataset)

        for i in range(5):
            print(dataset[i])

        plotSystemof2ndRang(dataset[0][0], dataset[0][1], dataset[0][2])

        sys = [reformatList(list(l[0])) for l in dataset]
        sol = [list(l[1]) for l in dataset]
        X = createX(sys, sol)
        Y = [l[2].astype("float32") for l in dataset]

        X = np.array([np.array(val) for val in X])
        Y = np.array([np.array(val) for val in Y])

        minMaxDataset(X,Y)

        plotMatricesLens(Y)

        scatterPoints(Y)

        scatterMeanSquaredError(Y)
