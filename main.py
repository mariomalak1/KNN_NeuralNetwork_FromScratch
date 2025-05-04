import LoadData
import Partitioning
import Normalization
import knn
import NeuralNetwork
import Evaluation

import numpy as np


if __name__ == "__main__":    
    loadData = LoadData.LoadData("../Kidney_Disease Data for classification.csv", 70)
    df = loadData.loadData()

    # make data without class label as notckd
    # df["classification"] = df["classification"].fillna("notckd")

    normalization = Normalization.Normalization(df)
    normalization.convertAllCategoricalFeaturesToNumeric()
    normalization.normalize()
    
    partitioning = Partitioning.Partitioning(normalization.df)


    x_train, y_train, x_test, y_test = partitioning.split(train_size=75)
    
    ### KNN
    # Create and train the classifier
    knn = knn.KNN(k=3)
    knn.fit(x_train.to_numpy(), y_train.to_numpy())
    
    # Make predictions
    predictions = knn.predict(x_test.to_numpy())

    # Test with known labels (for accuracy calculation)
    accuracy = Evaluation.Evaluation.accuracy(predictions, y_test)

    print(f"Accuracy: {accuracy * 100}%")



