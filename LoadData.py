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

