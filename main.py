from LoadData import LoadData 
from Partitioning import Partitioning
from Normalization import Normalization
from knn import KNN
from NeuralNetwork import NeuralNetwork
from Evaluation import Evaluation

import numpy as np


if __name__ == "__main__":    
    loadData = LoadData("../Kidney_Disease Data for classification.csv", 70)
    df = loadData.loadData()

    # remove "id" column
    df = df.drop('id', axis=1)

    # remove all nulls 
    df = df.dropna()

    # make data without class label as notckd
    # df["classification"] = df["classification"].fillna("notckd")

    normalization = Normalization(df)
    normalization.convertAllCategoricalFeaturesToNumeric()
    normalization.normalize()
    
    partitioning = Partitioning(normalization.df)


    x_train, y_train, x_test, y_test = partitioning.split(train_size=75)
    
    ## KNN
    knn = KNN(k=3)
    knn.fit(x_train.to_numpy(), y_train.to_numpy())
    
    # Make predictions
    predictions = knn.predict(x_test.to_numpy())

    # Test with known labels (for accuracy calculation)
    accuracy = Evaluation.accuracy(predictions, y_test)
    print(f"KNN - Accuracy: {accuracy * 100}%")


    ## ANN
    nn = NeuralNetwork(input_size=x_train.shape[1], hidden_size=2, output_size=1, learning_rate=0.005)
    nn.train(x_train.to_numpy(), y_train.to_numpy(), epochs=1000)

    predictions = nn.predict(x_test.to_numpy())

    # Test with known labels (for accuracy calculation)
    accuracy = Evaluation.accuracy(predictions, y_test)
    print(f"ANN - Accuracy: {accuracy * 100}%")
