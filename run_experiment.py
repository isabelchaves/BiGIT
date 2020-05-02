from typing import Dict

import pandas as pd

from configs.experiment_config import ExperimentConfig
from data.processing.feature_engineering import FeatureEngineering
from data.processing.textual_processing import PreProcessing
from experiments.tf_idf_experiments import TfIfdExperiments
from word_representations.word2vec import Word2VecExperiments


class RunExperiment:
    def __init__(self, data_path: str, language: Dict, vector_space: str, vector_method: str,
                 word_vectors_strategy: str, click_graph_interaction_number: int):
        self.data_path = data_path
        self.language = language
        self.vector_space = vector_space
        self.vector_method = vector_method
        self.word_vectors_strategy = word_vectors_strategy
        self.click_graph_interaction_number = click_graph_interaction_number

    def run(self):
        data = pd.read_csv(self.data_path + 'train.csv', encoding='latin-1')
        data = FeatureEngineering().run(data=data)
        language_process = PreProcessing(language=self.language)
        data['search_term_processed'] = data['search_term'].apply(lambda x: language_process.tokenizer(x))
        data['product_title_processed'] = data['product_title'].apply(lambda x: language_process.tokenizer(x))

        tfidf = TfIfdExperiments(vector_space=self.vector_space,
                                 vector_method=self.vector_method)

        # baseline_product_vs, baseline_query_vs = tfidf.run_baseline(data=data)

        clickgraph_product_vs, clickgraph_query_vs = \
            tfidf.run_with_click_graph(data=data, click_graph_interaction_number=self.click_graph_interaction_number)

        word2vec = Word2VecExperiments(word_vectors_strategy=self.word_vectors_strategy,
                                       vector_method=self.vector_method,
                                       vector_space=self.vector_space)

        # baseline_product_vs, baseline_query_vs = word2vec.run_baseline(data=data)

        # clickgraph_product_vs, clickgraph_query_vs = \
        #    word2vec.run_with_click_graph(data=data, click_graph_interaction_number=self.click_graph_interaction_number)

        print(data.head())


if __name__ == '__main__':
    experiment_config = ExperimentConfig()
    RunExperiment(data_path=experiment_config.data_path,
                  language=experiment_config.language,
                  vector_space=experiment_config.vector_space,
                  vector_method=experiment_config.vector_method,
                  word_vectors_strategy=experiment_config.word_vectors_strategy,
                  click_graph_interaction_number=experiment_config.click_graph_interaction_number).run()
