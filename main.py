import numpy as np
import pandas as pd


RND_SEED = 12345


class DatasetHandler:
    def __init__(self, csv_path):
        """Memorize the CSV file path"""
        self._path = csv_path
        self._data = None

    def load(self, preprocess=True):
        """Load dataset from the file and optionally do preprocessing."""
        self._data = pd.read_csv(self._path)
        if preprocess:
            self._preprocess()
        return self._data

    def train_test_split(self, train_ratio):
        """Split the dataset into train and test sets."""
        train_data = self._data.sample(frac=train_ratio, random_state=RND_SEED)
        test_data = self._data.drop(train_data.index)
        return (train_data, test_data)

    def _preprocess(self):
        """Remove bad samples, convert some datatypes, and normalize columns"""
        # Removing bad (incomplete) samples
        self._data.drop(
            self._data[
                (self._data['Glucose'] == 0) |
                (self._data['BloodPressure'] == 0) |
                (self._data['BMI'] == 0)
            ].index,
            inplace=True)

        # Splitting the dataset into X and Y
        x, y = self._data.iloc[:, :-1], self._data.iloc[:, -1]

        # Normalizing columns of X
        x = (x - x.min()) / (x.max() - x.min())

        # Mapping boolean values of Y to integer values {-1, 1}
        y.replace({False: -1, True: 1}, inplace=True)

        # Putting X and Y back together
        self._data = pd.concat([x, y], axis=1)


class Perceptron:
    def __init__(self, train_data, test_data):
        """Memorize training and test data and do some initializations."""
        n_train = train_data.shape[0]
        n_test = test_data.shape[0]

        # Add a column of all 1s to training and test data (for bias)
        train_data = np.hstack((np.ones((n_train, 1)), train_data))
        test_data = np.hstack((np.ones((n_test, 1)), test_data))

        # Split training data into X and Y
        self._x_train = train_data[:, :-1]
        self._y_train = train_data[:, -1]

        # Split test data into X and Y
        self._x_test = test_data[:, :-1]
        self._y_test = test_data[:, -1]

        # Initialize the weights vector
        d = self._x_train.shape[1]
        self._w = np.zeros((d, 1))

        # Keep the best weight vector (Pocket algorithm)
        self.w_hat = None

        # Store a sequence of model accuracy on test data
        self.accuracies = []

    def fit(self, epochs=1):
        for _ in range(epochs):
            for i, x in enumerate(self._x_train):
                y = self._y_train[i]
                y_p = self.predict(x)[0]
                if y != y_p:
                    self._evaluate_on_test()
                    self._w += (y * x.reshape(-1, 1))

        return (self.w_hat, self.accuracies)

    def predict(self, x):
        return np.sign(np.dot(x, self._w))

    def _evaluate_on_test(self):
        y_p = self.predict(self._x_test)
        accuracy = np.sum(y_p == self._y_test) / self._y_test.shape[0]
        if (len(self.accuracies) == 0) or (accuracy > max(self.accuracies)):
            self.w_hat = self._w
        self.accuracies.append(accuracy)


def main():
    dh = DatasetHandler('./assets/Dataset.csv')
    dh.load()
    train, test = dh.train_test_split(train_ratio=0.85)

    p = Perceptron(train, test)
    _, accuracies = p.fit()


if __name__ == '__main__':
    main()
