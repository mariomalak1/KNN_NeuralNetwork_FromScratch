from LoadData import LoadData 
from Partitioning import Partitioning
from Normalization import Normalization
from knn import KNN
from NeuralNetwork import NeuralNetwork
from Evaluation import Evaluation

import numpy as np

def preprocessing(fileName, percentageOfRowsToRead=70, train_set_size=70, knn_k=3, learning_rate=0.005, hidden_layers=3, thresholdAccuracy=0.8):
    loadData = LoadData(fileName, percentageOfRowsToRead)
    df = loadData.loadData()

    # remove "id" column
    df = df.drop('id', axis=1)

    # remove all nulls 
    df = df.dropna()

    # shuffle the data
    df = df.sample(frac = 1)

    # make data without class label as notckd
    # df["classification"] = df["classification"].fillna("notckd")

    normalization = Normalization(df)
    normalization.convertAllCategoricalFeaturesToNumeric()
    normalization.normalize()
    
    partitioning = Partitioning(normalization.df)


    x_train, y_train, x_test, y_test = partitioning.split(train_size=train_set_size)
    
    ## KNN
    knn_accuracy = 0
    ann_accuracy = 0
    trainingCounter = 0
    
    while knn_accuracy < thresholdAccuracy or ann_accuracy < thresholdAccuracy:
        trainingCounter += 1

        if knn_accuracy < thresholdAccuracy:
            knn = KNN(k=knn_k)
            knn.fit(x_train.to_numpy(), y_train.to_numpy())
            
            # Make predictions
            predictions = knn.predict(x_test.to_numpy())

            # Test with known labels (for accuracy calculation)
            knn_accuracy = Evaluation.accuracy(predictions, y_test)
            print(f"training Number {trainingCounter} for KNN - Accuracy: {knn_accuracy * 100}%")

        if ann_accuracy < thresholdAccuracy:
            ## ANN
            nn = NeuralNetwork(input_size=x_train.shape[1], hidden_size=hidden_layers, output_size=1, learning_rate=learning_rate)
            nn.train(x_train.to_numpy(), y_train.to_numpy(), epochs=1000)

            predictions = nn.predict(x_test.to_numpy())

            # Test with known labels (for accuracy calculation)
            ann_accuracy = Evaluation.accuracy(predictions, y_test)
            print(f"training Number {trainingCounter} for ANN - Accuracy: {ann_accuracy * 100}%")

        # force stop condition 
        if trainingCounter > 30:
            break

    return knn_accuracy, ann_accuracy