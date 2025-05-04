import pandas as pd 

class LoadData:
    defaultPercentage = 20

    def __init__(self, file_path, percentage_num_of_rows):
        self.file_path = file_path

       # validate percentage_num_of_rows
        try:
            percentage_num_of_rows = float(percentage_num_of_rows)
            if(percentage_num_of_rows < 0 or percentage_num_of_rows > 100):
                percentage_num_of_rows = LoadData.defaultPercentage
        except:
            print("invalid number, will take default")
            percentage_num_of_rows = LoadData.defaultPercentage

        self.percentage_num_of_rows = percentage_num_of_rows / 100

        rows = pd.read_csv(self.file_path, usecols=[0]).shape[0]

        self.numOfRows = rows * self.percentage_num_of_rows
        


    def loadData(self):
        try:
            self.data = pd.read_csv(self.file_path, nrows=self.numOfRows)           
            return self.data
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []


    def partition(self, train_size=0.8):
        # if the user will take 100% from reading the data
        if self.percentage_num_of_rows == 1:
            train_end = 280

            train_data = self.data.iloc[:train_end].reset_index(drop=True)
            test_data = self.data.iloc[train_end:].reset_index(drop=True)

        else:
            train_end = int(len(self.data) * train_size)

            train_data = self.data.iloc[:train_end].reset_index(drop=True)
            test_data = self.data.iloc[train_end:].reset_index(drop=True)

        # Last column is the label
        y_train = train_data.iloc[:, -1]
        x_train = train_data.iloc[:, :-1]

        y_test = test_data.iloc[:, -1]
        x_test = test_data.iloc[:, :-1]

        return x_train, y_train, x_test, y_test