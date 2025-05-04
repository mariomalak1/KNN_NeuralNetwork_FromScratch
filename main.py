import LoadData
import Partitioning
import knn
import Normalization

import numpy as np


if __name__ == "__main__":

    loadData = LoadData.LoadData("../Kidney_Disease Data for classification.csv", 100)
    df = loadData.loadData()

    # make data without class label as notckd
    # df["classification"] = df["classification"].fillna("notckd")

    normalization = Normalization.Normalization(df)
    normalization.convertAllCategoricalFeaturesToNumeric()
    normalization.normalize()
    
    partitioning = Partitioning.Partitioning(normalization.df)

    x_train, y_train, x_test, y_test = partitioning.split()

    # # Sample dataset (features and labels)
    # X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # y_train = np.array([0, 0, 1, 1, 0, 1])
    
    # Create and train the classifier
    knn = knn.KNN(k=3)
    knn.fit(x_train.to_numpy(), y_train.to_numpy())
    
    # # Test data
    # X_test = np.array([[1, 1.5], [8, 9], [0, 3], [5, 4]])
    
    # Make predictions
    predictions = knn.predict(x_test.to_numpy())

    # Test with known labels (for accuracy calculation)
    accuracy = knn.score(predictions, y_test)

    print(f"Accuracy: {accuracy * 100}%")

