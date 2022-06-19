import csv
import random
import numpy as np

# Numarul de sisteme generate
_GENERATE_SYSTEM_NUMBER = 1000000000

# Valoare maxima pentru coeficientii matricei, necunoscute.
_MAX = 1

# In cazul alegerii metodei de a stoca sisteme de rang diferit in acelasi set de date: Selectarea rang minim si maxim.
_MIN_RANK = 2
_MAX_RANK = 20

# In cazul alegerii metodei de a stoca doar sisteme de acelasi in setul de date: Selectarea rangului unic
_RANK = 100

# Boolean True: Metoda de a stoca mai multe sisteme de rang diferit in acelasi set de date
# Boolean False: Metoda de a stoca doar sisteme de acelasi rang in setul de date.
_MULTI_RANK = False


def getDatasetPath(boolean):
    if boolean:
        datasetPath = "Neural Network - Systems with Various ranks/datasets/"
    else:
        datasetPath = "Neural Network - Systems with a Single Rank/datasets/"

    return datasetPath


def multiUniRankSelector(boolean):
    if boolean:
        systemSize = random.randrange(_MIN_RANK, _MAX_RANK)
    else:
        systemSize = _RANK

    return systemSize


def getSystem():
    """
    Se genereaza valori aleatorii pentru coeficientii independenti si dependenti.
    Valoarea _Max reprezinta valoarea maxima pe care o paote lua oricare coeficient al sistemelor, respectiv
    necunoscutelor. In cazul de fata se vor genera sisteme de [2x2] -> pana la [5x5]
    :returns: coeficienti dependenti aleatori, coeficienti independenti aleatori
    """
    systemSize = multiUniRankSelector(_MULTI_RANK)
    syst = np.round(np.random.uniform(0, _MAX, [systemSize, systemSize]), 18)
    sol = np.round(np.random.uniform(0, _MAX, [systemSize, 1]), 18)
    return syst, sol


def getOutput(syst, sol):
    """
    Se apleaza functia anterioara 'getSystem()' pentru a genera sisteme cat timp determinantul sistemului este 0.
    Daca determinantul sistemului este 0, acesta nu are solutii unice.
    :returns un array cu valorile necunoscutelor sistemului -> solutiile sistemului
    """
    while np.linalg.det(syst) == 0:
        syst, sol = getSystem()
    return np.linalg.solve(syst, sol)


def generateEntries(x):
    """
    Se apeleaza functiile "getSystem()" si "getOutput()" si se retin valorile sistemului intr-o baza de date de tip
    csv. :param x: numarul de generari ale coeficientilor dependenti si independenti (sistemelor) :return: scrierea
    in fisiere a coeficientilor dependenti / coeficientilor independenti / necunoscutelor -> transformate in bytecode
    in format csv.
    """
    for i in range(x):
        syst, sol = getSystem()
        out = getOutput(syst, sol)
        with open(getDatasetPath(_MULTI_RANK) + 'Systsems100.csv', 'a+', newline='',
                  encoding='UTF8') as f:
            writer = csv.writer(f)
            #writer.writerow(['system','solution','output'])
            if np.max(out) < _MAX and np.min(out) > -_MAX and np.max(syst) < _MAX and np.min(syst) > 0 and np.max(
                    sol) < _MAX and np.min(sol) > 0:
                writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
                print(out)
                print(len(out))
            f.close()


# Generarea sistemelor
if __name__ == '__main__':
    generateEntries(1000000000)
