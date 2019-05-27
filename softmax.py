import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_values = map(lambda x: np.exp(x), L)
    exp_values = list(exp_values)
    total = sum(exp_values)
    soft_max = map(lambda x: x / total, exp_values)

    return list(soft_max)


print(softmax([2, 1, 0]))