import warnings
from copy import deepcopy
from typing import Dict

import pandas as pd

from src.configs.experiment_config import ExperimentConfig
from src.configs.variables_const import VariablesConsts
from src.data.processing.feature_engineering import FeatureEngineering
from src.data.processing.textual_processing import PreProcessing
from src.predictions import Predict
from src.word_representations.bert import BertExperiments
from src.word_representations.tf_idf import TfIfdExperiments
from src.word_representations.word2vec import Word2VecExperiments

warnings.filterwarnings('ignore')


class RunExperiment:
    def __init__(self, data_path: str, language: Dict, vector_space: str, vector_method: str,
                 word_vectors_strategy: str, bigit_interaction_number: int, bigit_initialization: str):
        self.data_path = data_path
        self.language = language
        self.vector_space = vector_space
        self.vector_method = vector_method
        self.word_vectors_strategy = word_vectors_strategy
        self.bigit_interaction_number = bigit_interaction_number
        self.bigit_initialization = bigit_initialization

    def run(self):
        data = pd.read_parquet(self.data_path + 'train_set.parquet')
        data = FeatureEngineering().run(data=data)
        language_process = PreProcessing(language=self.language)
        data[VariablesConsts.SEARCH_TERM_PROCESSED] = data[VariablesConsts.SEARCH_TERM].apply(
            lambda x: language_process.tokenizer(x))
        data[VariablesConsts.PRODUCT_TITLE_PROCESSED] = data[VariablesConsts.PRODUCT_TITLE].apply(
            lambda x: language_process.tokenizer(x))
        data = data[data['search_term_processed'] != '']
        data.reset_index(drop=True, inplace=True)
        product_ids = dict(enumerate(data[VariablesConsts.PRODUCT_ID].unique()))

        # TF-IDF #
        tfidf = TfIfdExperiments(vector_space=self.vector_space,
                                 vector_method=self.vector_method,
                                 product_ids=deepcopy(product_ids))

        tfidf_product_vs, tfidf_query_vs = tfidf.run_baseline(data=deepcopy(data))

        Predict(model_class=deepcopy(tfidf),
                product_vector_space=tfidf_product_vs,
                search_terms_vector_space=tfidf_query_vs,
                product_ids=deepcopy(product_ids)).run_predictions()

        tfidf_graph_product_vs, tfidf_graph_query_vs = \
            tfidf.run_with_bigit(data=deepcopy(data),
                                 bigit_interaction_number=self.bigit_interaction_number,
                                 bigit_initialization=self.bigit_initialization)

        Predict(model_class=deepcopy(tfidf),
                product_vector_space=tfidf_graph_product_vs,
                search_terms_vector_space=tfidf_graph_query_vs,
                product_ids=deepcopy(product_ids)).run_predictions()

        # Word2vec #
        word2vec = Word2VecExperiments(word_vectors_strategy=self.word_vectors_strategy,
                                       vector_method=self.vector_method,
                                       vector_space=self.vector_space,
                                       product_ids=deepcopy(product_ids))

        word2vec_product_vs, word2vec_queries_vs = word2vec.run_baseline(data=deepcopy(data))
        Predict(model_class=deepcopy(word2vec),
                product_vector_space=word2vec_product_vs,
                search_terms_vector_space=word2vec_queries_vs,
                product_ids=deepcopy(product_ids)).run_predictions()

        word2vec_graph_product_vs, word2vec_graph_queries_vs = \
            word2vec.run_with_bigit(data=deepcopy(data),
                                    bigit_interaction_number=self.bigit_interaction_number,
                                    bigit_initialization=self.bigit_initialization)

        Predict(model_class=deepcopy(word2vec),
                product_vector_space=word2vec_graph_product_vs,
                search_terms_vector_space=word2vec_graph_queries_vs,
                product_ids=deepcopy(product_ids)).run_predictions()

        # BERT #
        bert = BertExperiments(word_vectors_strategy=self.word_vectors_strategy,
                               vector_method=self.vector_method,
                               vector_space=self.vector_space,
                               product_ids=deepcopy(product_ids))

        bert_product_vs, bert_queries_vs = bert.run_baseline(data=deepcopy(data))
        Predict(model_class=deepcopy(bert),
                product_vector_space=bert_product_vs,
                search_terms_vector_space=bert_queries_vs,
                product_ids=deepcopy(product_ids)).run_predictions()

        bert_graph_product_vs, bert_graph_queries_vs = \
            bert.run_with_bigit(data=deepcopy(data),
                                bigit_interaction_number=self.bigit_interaction_number,
                                bigit_initialization=self.bigit_initialization)

        Predict(model_class=deepcopy(bert),
                product_vector_space=bert_graph_product_vs,
                search_terms_vector_space=bert_graph_queries_vs,
                product_ids=deepcopy(product_ids)).run_predictions()

        print(data.head())


if __name__ == '__main__':
    experiment_config = ExperimentConfig()
    RunExperiment(data_path=experiment_config.data_path,
                  language=experiment_config.language,
                  vector_space=experiment_config.vector_space,
                  vector_method=experiment_config.vector_method,
                  word_vectors_strategy=experiment_config.word_vectors_strategy,
                  bigit_interaction_number=experiment_config.bigit_interaction_number,
                  bigit_initialization=experiment_config.bigit_initialization).run()
