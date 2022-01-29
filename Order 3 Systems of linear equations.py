import csv
import random
import numpy as np
from scipy.linalg import solve

_MAX = 100

def getSystem():
  systemSize = 3
  syst = np.round(np.random.uniform(-_MAX-1, _MAX, [systemSize, systemSize]),2)
  print("System: {} ".format(syst))
  sol = np.round(np.random.uniform(-_MAX-1, _MAX, [systemSize, 1]),2)
  print("SOLUTIONS: {} ".format(sol))
  return syst, sol

def getOutput(syst, sol):
  while np.linalg.det(syst) == 0:
    syst, sol = getSystem()
  return np.round(solve(syst, sol),2)

def generateEntries(x):
  for i in range(x):
    syst, sol = getSystem()
    out = getOutput(syst, sol)
    with open('C:\\Users\\Adrian\\Desktop\\GenerateLicenta\\licenta_dataset3Ord.csv', 'a+', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        #writer.writerow(['system','solution','output'])
        writer.writerow([str(syst.tobytes()), str(sol.tobytes()), str(out.tobytes())])
        print("Solved Values X,Y,Z: {} ".format(out))
        f.close()

generateEntries(1000000)