import pandas as pd

def testingDataWithPrediction(predictions, x_test, y_test):
    predicted_labels = pd.Series(predictions, index=x_test.index, name='Predicted')
    predicted_labels = predicted_labels.map({0.0: "ckd", 1.0: "notckd"})
    actual_labels = pd.Series(y_test.values, name='Actual')
    actual_labels = actual_labels.map({0.0: "ckd", 1.0: "notckd"})
    
    result_df = pd.concat([
        x_test.reset_index(drop=True),
        predicted_labels.reset_index(drop=True),
        actual_labels.reset_index(drop=True)
    ], axis=1)
    
    records = [result_df.columns.tolist()] + result_df.values.tolist()
    
    return records