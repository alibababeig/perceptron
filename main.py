import pandas as pd


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


def main():
    dh = DatasetHandler('./assets/Dataset.csv')
    dataset = dh.load()
    dataset.to_csv('./foo.csv')


if __name__ == '__main__':
    main()
