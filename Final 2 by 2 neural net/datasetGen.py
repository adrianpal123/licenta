import csv
import numpy as np
from scipy.linalg import solve

_MAX = 1000000


def getSystem():
    """
      Se genereaza valori aleatorii pentru coeficientii independenti si dependenti.
      Valoarea _Max este de 1 milion, coficientilor, respectiv necunoscutelor sistemului trebuie sa fie in intervalul [-_MAX,_MAX].
      :returns: coeficienti dependenti aleatori, coeficienti independenti aleatori
      """
    systemSize = 2
    syst = np.round(np.random.uniform(-1, _MAX/np.random.randint(1,10000), [systemSize, systemSize]), 8)
    sol = np.round(np.random.uniform(-1, _MAX/np.random.randint(1,10000), [systemSize, 1]), 8)
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
    counter = 0
    for i in range(x):
        syst, sol = getSystem()
        out = getOutput(syst, sol)
        with open('Datasets/CosineSimilarity[-1000000,1000000]Dataset.csv', 'a+', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            #writer.writerow(['system','solution','output'])
            if np.max(out) < _MAX and np.min(out) > -_MAX and np.max(syst) < _MAX and np.min(syst) > -_MAX and np.max(
                    sol) < _MAX and np.min(sol) > -_MAX:
                writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
                print("Coeficientii Dependenti: {} ".format(syst) + '\n' + "Coeficientii Independenti: {} ".format(
                    sol) + '\n' + "Necunoscutele [X si Y]: {} ".format(out) + '\n' + "Counter = " + str(
                    counter))
                counter += 1
                print(str(syst) + '\n' + str(sol) + '\n' + str(out) + '\n' + "Counter = " + str(counter))

        f.close()


if __name__ == '__main__':
    generateEntries(1000000000000000)
