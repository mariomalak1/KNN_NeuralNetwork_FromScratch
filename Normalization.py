class Normalization:
    def __init__(self, data):
        self.df = data.copy()

    def convertAllCategoricalFeaturesToNumeric(self):
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].astype('category').cat.codes


    def normalize(self):
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        self.df[numeric_cols] = (self.df[numeric_cols] - self.df[numeric_cols].min()) / (self.df[numeric_cols].max() - self.df[numeric_cols].min())

