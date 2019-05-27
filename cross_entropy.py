import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    values = map(lambda y, p: y * np.log(p) + (1 - y) * np.log(1 - p), Y, P)

    return - sum(values)