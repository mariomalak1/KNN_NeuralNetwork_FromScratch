import pandas as pd

def testingDataWithPrediction(predictions, x_test, y_test):
    # Convert predictions to a pandas Series if needed
    predicted_labels = pd.Series(predictions, index=x_test.index, name='Predicted')

    # Concatenate the test features with the predicted label
    x_test_with_predictions = pd.concat([x_test.reset_index(drop=True), predicted_labels.reset_index(drop=True)], axis=1)

    # Optional: Include actual label for comparison
    actual_labels = pd.Series(y_test.values, name='Actual')
    x_test_with_predictions['Actual'] = actual_labels

    # Convert to list of lists (e.g., for displaying in the GUI)
    records = [x_test_with_predictions.columns.tolist()] + x_test_with_predictions.values.tolist()

    return records