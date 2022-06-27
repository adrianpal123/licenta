import csv
import numpy as np
import tensorflow as tf
import time

dimensions = (30, 30)
Matrix = np.zeros(dimensions)
MatrixFree = np.zeros(30)

with open('Reservoir Non zeros - Matrix.csv', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for i, line in enumerate(reader):
        print(i, line)
        if line[2] == '':
            Matrix[int(line[0]) - 1][int(line[1]) - 1] = float(line[3])
        else:
            Matrix[int(line[0]) - 1][int(line[1]) - 1] = float(line[2])

print(Matrix)

new_Matrix = []
for line in Matrix:
    line1 = np.append(line, 0)
    new_Matrix.append(line1)

for line in new_Matrix:
    print(len(line))

print(new_Matrix)

reconstructed_model = tf.keras.models.load_model("neural network models results/saved neural network")

q = reconstructed_model.predict(np.array([new_Matrix, ]))

start = time.perf_counter()
print("PREDICTED")
print(q)
end = time.perf_counter()
print("Time for prediction:")
print(f"System solved  in {end - start:0.9f} seconds")

start = time.perf_counter()
print("SOLVED VALUES")
a = [0] * 30
print(Matrix)
print(np.linalg.solve(Matrix, a))
end = time.perf_counter()
print("Time for solving:")
print(f"System solved  in {end - start:0.9f} seconds")
