import csv
import random
import numpy as np

_MAX = 1


# Generarea sistemelor de ecuații liniare, rangul apartinand intervalului [2,10]
def getSystem():
    """
    Se genereaza valori aleatorii pentru coeficientii independenti si dependenti.
    Valoarea _Max reprezinta valoarea maxima pe care o paote lua oricare coeficient al sistemelor, respectiv
    necunoscutelor. In cazul de fata se vor genera sisteme de [2x2] -> pana la [5x5]
    :returns: coeficienti dependenti aleatori, coeficienti independenti aleatori
    """
    random.randrange(2, 5)
    systemSize = 5
    syst = np.round(np.random.uniform(0, _MAX, [systemSize, systemSize]), 18)
    sol = np.round(np.random.uniform(0, _MAX, [systemSize, 1]), 18)
    return syst, sol


# Rezolvarea sistemelor folosind metoda "solve" din libraria numpy.
# Metoda solve se foloseste de metoda directa Cramer pentru a rezolva aceste sisteme.
def getOutput(syst, sol):
    while np.linalg.det(syst) == 0:
        syst, sol = getSystem()
    return np.linalg.solve(syst, sol)


# Generarea intrarilor, aplicandu-se conditiile initiale
def generateEntries(x):
    _5 = 0
    _4 = 0
    _3 = 0
    _2 = 0
    _NUMBER = 25000

    for i in range(x):
        syst, sol = getSystem()
        out = getOutput(syst, sol)
        with open('datasets/FloatValueInterval[0,1](18-Decimals)-Size[2-5]Dataset.csv', 'a+', newline='',
                  encoding='UTF8') as f:
            writer = csv.writer(f)
            # writer.writerow(['system','solution','output'])
            if np.max(out) < _MAX and np.min(out) > 0 and np.max(syst) < _MAX and np.min(syst) > 0 and np.max(
                    sol) < _MAX and np.min(sol) > 0:
                if len(out) == 2 and _2 < _NUMBER:
                    writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
                    _2 = _2 + 1
                if len(out) == 3 and _3 < _NUMBER:
                    writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
                    _3 = _3 + 1
                if len(out) == 4 and _4 < _NUMBER:
                    writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
                    _4 = _4 + 1
                if len(out) == 5 and _5 < _NUMBER:
                    writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
                    _5 = _5 + 1
                print(syst)
                if _2 >= _NUMBER or _3 >= _NUMBER or _4 >= _NUMBER or _5 >= _NUMBER:
                    break
            f.close()


# Generarea a unui milion de astfel de sisteme
generateEntries(1000000000000000)
