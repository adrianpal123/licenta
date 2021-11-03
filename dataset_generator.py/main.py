import csv
import random
import numpy as np
from scipy.linalg import solve

_MAX = 100

def getSystem():
  systemSize = random.randrange(2, 6)
  syst = np.random.uniform(-_MAX-1, _MAX, [systemSize, systemSize])
  sol = np.random.uniform(-_MAX-1, _MAX, [systemSize, 1])
  return syst, sol

def getOutput(syst, sol):
  while np.linalg.det(syst) == 0:
    syst, sol = getSystem()
  return solve(syst, sol)

def generateEntries(x):
  for i in range(x):
    syst, sol = getSystem()
    out = getOutput(syst, sol)
    with open('C:\\Users\\Adrian\\Desktop\\Lucrare licenta Inf\\licenta_dataset.csv', 'a', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        #writer.writerow(['system','solution','output'])
        writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
        f.close()

generateEntries(99)
