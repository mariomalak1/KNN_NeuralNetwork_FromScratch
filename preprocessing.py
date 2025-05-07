from LoadData import LoadData 
from Partitioning import Partitioning
from Normalization import Normalization
from knn import KNN
from NeuralNetwork import NeuralNetwork
from Evaluation import Evaluation

import numpy as np

def preprocessing(fileName, percentageOfRowsToRead=70, train_set_size=75, knn_k=3, learning_rate=0.005, hidden_layers=3, thresholdAccuracy=90):
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
    knn = KNN(k=knn_k)
    knn.fit(x_train.to_numpy(), y_train.to_numpy())
    
    # Make predictions
    predictions = knn.predict(x_test.to_numpy())

    # Test with known labels (for accuracy calculation)
    knn_accuracy = Evaluation.accuracy(predictions, y_test)
    print(f"KNN - Accuracy: {knn_accuracy * 100}%")


    ## ANN
    nn = NeuralNetwork(input_size=x_train.shape[1], hidden_size=hidden_layers, output_size=1, learning_rate=learning_rate)
    nn.train(x_train.to_numpy(), y_train.to_numpy(), epochs=1000)

    predictions = nn.predict(x_test.to_numpy())

    # Test with known labels (for accuracy calculation)
    ann_accuracy = Evaluation.accuracy(predictions, y_test)
    print(f"ANN - Accuracy: {ann_accuracy * 100}%")

    # predict new data that entered by user

    # newDataPredict = None

    # newDataEntered=[48.0,80.0,1.02,1.0,0.0,,normal,notpresent,notpresent,121.0,36.0,1.2,,,15.4,44,7800,5.2,yes,yes,no,good,no,no]
    # # for predict new record 
    # if newDataEntered:
    #     isAllDataEntered = True
    #     for i in newDataEntered:
    #         if not i:
    #             isAllDataEntered = False
    #     if isAllDataEntered:
    #         newDataEntered = np.array(newDataEntered)

    #         if(knn_accuracy < thresholdAccuracy and ann_accuracy < thresholdAccuracy):
    #             newDataPredict = nn.predict(newDataEntered)
    #         elif (knn_accuracy < thresholdAccuracy):
    #             newDataPredict = nn.predict(newDataEntered)
    #         elif (ann_accuracy < thresholdAccuracy):
    #             newDataPredict = knn.predict(newDataEntered)
    #         else:
    #             newDataPredict = nn.predict(newDataEntered)

    #         if newDataPredict == 0:
    #             newDataPredict = "notckd"
    #         elif newDataPredict == 1:
    #             newDataPredict = "ckd"

    return knn_accuracy, ann_accuracy