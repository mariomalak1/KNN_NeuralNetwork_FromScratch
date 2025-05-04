import numpy as np

class Evaluation:
    def accuracy(predictions, y):
        return np.mean(predictions == y)
    