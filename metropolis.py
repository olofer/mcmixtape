import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="metropolis.txt")
args = parser.parse_args()

X = np.loadtxt(args.file)
print(X.shape)

mu = np.mean(X, axis=0)
print("mean:", mu)
median = np.median(X, axis=0)
print("median:", median)
print("#both should be:",1.0)
print("stdev:", np.std(X, axis=0))
print("#should be:",np.sqrt(2))
print("MAD:", np.median(np.abs(X - np.outer(np.ones(X.shape[0]), median)), axis=0))
print("#should be:",np.log(2))

print("covariance matrix (should be 2*I):")
Xtilde = X - np.outer(np.ones(X.shape[0]), mu)
C = (Xtilde.T @ Xtilde) / (X.shape[0] - 1)
print(C)
