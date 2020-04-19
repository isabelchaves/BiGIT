from random import randint

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineering:
    def __init__(self):
        self.max_clicks = 200
        self.min_clicks = 10

    def _create_click_score(self, data):
        max_relevance = max(list(data.relevance.unique()))

        data['clicks'] = data['relevance'].apply(
            lambda x: randint(self.min_clicks, self.max_clicks) if x == max_relevance else randint(0, self.min_clicks))

        data['click_score'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.relevance.values.reshape(-1, 1))

        data['click_score'] = data.apply(
            lambda x: (1 / (self.max_clicks * x['click_score'])) * x['clicks'] if x.click_score else 0, axis=1)

        return data

    def run(self, data: pd.DataFrame):
        data = self._create_click_score(data=data)
        return data
