import pandas as pd


class DataSplitter:
    def __init__(self, file_path: str = "hdb_price_prediction.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

    def split(self):
        train = self.data[
            self.data["year"].between(2017, 2020)
        ].copy()

        validation = self.data[
            self.data["year"] == 2021
        ].copy()

        test = self.data[
            self.data["year"] == 2022
        ].copy()

        return train, validation, test
