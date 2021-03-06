import random
from random import randint

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.configs.variables_const import VariablesConsts


class FeatureEngineering:
    def __init__(self):
        self.max_clicks = 100
        self.min_clicks = 10

    def _create_click_score(self, data):
        relevance_threshold = data[VariablesConsts.RELEVANCE].quantile(0.60)

        random.seed(42)
        data[VariablesConsts.CLICKS] = data[VariablesConsts.RELEVANCE].apply(
            lambda x: randint(self.min_clicks, self.max_clicks) if x > relevance_threshold else randint(0, self.min_clicks))

        data[VariablesConsts.CLICK_SCORE] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[[VariablesConsts.CLICKS]])

        # data[VariablesConsts.CLICK_SCORE] = data.apply(
        #     lambda x: (1 / (self.max_clicks * x[VariablesConsts.CLICK_SCORE])) * x[
        #         VariablesConsts.CLICKS] if x.click_score else 0, axis=1)

        return data

    def run(self, data: pd.DataFrame):
        data = self._create_click_score(data=data)
        return data
