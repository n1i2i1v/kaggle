import numpy as np


def loss(y, predicted):
    return np.sqrt(sum((np.log(y+1) - np.log(predicted+1))**2)/len(y))
