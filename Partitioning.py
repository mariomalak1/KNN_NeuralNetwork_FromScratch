class Partitioning:
    def __init__(self, data):
        self.data = data

    def split(self, train_size=80):
        if not 0 < train_size < 100:
            raise ValueError("train_size must be a float between 0 and 100")

        # Shuffle the dataset randomly
        shuffled_data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Convert percentage to proportion
        train_size /= 100

        train_end = int(len(shuffled_data) * train_size)

        train_data = shuffled_data.iloc[:train_end]
        test_data = shuffled_data.iloc[train_end:]

        # Last column is the label
        y_train = train_data.iloc[:, -1]
        x_train = train_data.iloc[:, :-1]

        y_test = test_data.iloc[:, -1]
        x_test = test_data.iloc[:, :-1]

        return x_train, y_train, x_test, y_test
