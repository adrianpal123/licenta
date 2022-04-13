import csv
import numpy as np
from scipy.linalg import solve

_MAX = 1


def getSystem():
    """
      Se genereaza valori aleatorii pentru coeficientii independenti si dependenti.
      Valoarea _Max este de 1 milion, coficientilor, respectiv necunoscutelor sistemului trebuie sa fie in intervalul [-_MAX,_MAX].
      :returns: coeficienti dependenti aleatori, coeficienti independenti aleatori
      """
    systemSize = 2
    syst = np.round(np.random.uniform(-_MAX, _MAX, [systemSize, systemSize]), 8)
    sol = np.round(np.random.uniform(-_MAX, _MAX, [systemSize, 1]), 8)
    return syst, sol


def getOutput(syst, sol):
    """
    Se apleaza functia anterioara 'getSystem()' pentru a genera sisteme cat timp determinantul sistemului este 0.
    Daca determinantul sistemului este 0, acesta nu are solutii unice.
    :returns un array cu valorile necunoscutelor sistemului -> solutiile sistemului
    """
    while np.linalg.det(syst) == 0:
        syst, sol = getSystem()
    return solve(syst, sol)


def generateEntries(x):
    """
    Se apeleaza functiile "getSystem()" si "getOutput()" si se retin valorile sistemului intr-o baza de date de tip csv.
    :param x: numarul de generari ale coeficientilor dependenti si independenti (sistemelor)
    :return: scrierea in fisiere a coeficientilor dependenti / coeficientilor independenti / necunoscutelor -> transformate
           in bytecode in format csv.
    """
    maxim = 0
    maximCoefD = 0
    maximCoefI = 0
    counter = 0
    for i in range(x):
        syst, sol = getSystem()
        out = getOutput(syst, sol)
        with open('Datasets/CosineSimilarityDataset.csv', 'a+', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            #writer.writerow(['system','solution','output'])
            if np.max(out) < 1 and np.min(out) > 0 and np.max(syst) < _MAX and np.min(syst) > 0 and np.max(
                    sol) < _MAX and np.min(sol) > 0:
                writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
                print("Coeficientii Dependenti: {} ".format(syst) + '\n' + "Coeficientii Independenti: {} ".format(
                    sol) + '\n' + "Necunoscutele [X,Y,Z,W,A,R..ETC]: {} ".format(out) + '\n' + "Counter = " + str(
                    counter))
                print(syst)
                print(sol)
                print(out)
                if np.max(out) > maxim:
                    maxim = np.max(out)
                if np.max(sol) > maximCoefI:
                    maximCoefI = np.max(sol)
                if np.max(syst) > maximCoefD:
                    maximCoefD = np.max(syst)
                print(len(out))
            f.close()

        print("MAXIM Solutii: ")
        print(maxim)
        print("Maxim Coef Independenti: ")
        print(maximCoefI)
        print("Maxim Coef Dependenti: ")
        print(maximCoefI)
        print("Counter = " + str(counter))


if __name__ == '__main__':
    generateEntries(10000000000)
