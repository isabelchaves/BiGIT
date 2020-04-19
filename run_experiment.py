from typing import Dict

import pandas as pd
from configs.experiment_config import ExperimentConfig
from data.processing.feature_engineering import FeatureEngineering
from data.processing.textual_processing import PreProcessing
from experiments.tf_idf_experiments import TfIfdExperiments


class RunExperiment:
    def __init__(self, data_path: str, language: Dict, vector_space: str, vector_method: str):
        self.data_path = data_path
        self.language = language
        self.vector_space = vector_space
        self.vector_method = vector_method

    def run(self):
        data = pd.read_csv(self.data_path + 'train.csv', encoding='latin-1')
        data = FeatureEngineering().run(data=data)
        language_process = PreProcessing(language=self.language)
        data['search_term_processed'] = data['search_term'].apply(lambda x: language_process.tokenizer(x))
        data['product_title_processed'] = data['product_title'].apply(lambda x: language_process.tokenizer(x))

        TfIfdExperiments(vector_space=self.vector_space,
                         vector_method=self.vector_method).run(data=data)

        print(data.head())


if __name__ == '__main__':
    experiment_config = ExperimentConfig()
    RunExperiment(data_path=experiment_config.data_path,
                  language=experiment_config.language,
                  vector_space=experiment_config.vector_space,
                  vector_method=experiment_config.vector_method).run()

