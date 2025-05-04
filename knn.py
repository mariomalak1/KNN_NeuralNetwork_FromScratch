import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            distances = []
            for x_train in self.X_train:
                distances.append(self.euclidean_distance(x_test, x_train))
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            prediction = np.argmax(np.bincount(nearest_labels))
            predictions.append(prediction)
        return np.array(predictions)

 
