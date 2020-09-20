import numpy as np
import pandas as pd
from tqdm import tqdm

from src.configs.experiment_config import ExperimentConfig
from src.configs.variables_const import VariablesConsts
from src.data.processing.textual_processing import PreProcessing
from src.evaluation.evaluation_method import EvaluationMethod


class Predict:

    def __init__(self, model_class, product_vector_space, search_terms_vector_space, product_ids):
        self.model_class = model_class
        self.product_vector_space = product_vector_space
        self.search_terms_vector_space = search_terms_vector_space
        self.product_ids = product_ids

    def run_predictions(self, ):
        data_to_predict = pd.read_parquet(ExperimentConfig.data_path + 'test_set.parquet')
        language_process = PreProcessing(language=ExperimentConfig.language)
        data_to_predict[VariablesConsts.SEARCH_TERM_PROCESSED] = data_to_predict[VariablesConsts.SEARCH_TERM].apply(
            lambda x: language_process.tokenizer(x))
        data_to_predict[VariablesConsts.PRODUCT_TITLE_PROCESSED] = data_to_predict[VariablesConsts.PRODUCT_TITLE].apply(
            lambda x: language_process.tokenizer(x))

        products, queries = self.model_class.prepare_data(data=data_to_predict)

        # queries = self._approximate_queries_the_vector_space(queries=queries)

        EvaluationMethod(product_ids=self.product_ids).run(data=data_to_predict,
                                                           data_to_evaluate=queries,
                                                           vector_space_to_search=self.product_vector_space,
                                                           evaluate_column=VariablesConsts.SEARCH_TERM_PROCESSED)

    def _approximate_queries_the_vector_space(self, queries):
        similar_queries = pd.DataFrame(columns=[VariablesConsts.DISTANCE, VariablesConsts.SEARCH_TERM_PROCESSED])

        for value in tqdm(queries, desc='Evaluate the products of the queries'):
            ids, distance = self.search_terms_vector_space.knnQuery(queries[value], k=10)
            # print(query)
            similar_queries = similar_queries.append(
                pd.DataFrame([[x[1]] + [self.search_terms_vector_space[x[0]]] for x in zip(ids, distance)],
                             columns=[VariablesConsts.DISTANCE, VariablesConsts.SEARCH_TERM_PROCESSED]),
                ignore_index=True)

            similar = np.stack(
                similar_queries.apply(lambda x: [x.distance * value for value in x.search_term_processed], axis=1).to_numpy())
            similar = similar.sum(axis=0, dtype='float') / sum(similar_queries.distance)
            update = np.add(queries[value], similar)
            queries[value] = update / 10
            # queries[value] = similar.sum(axis=0, dtype='float') / 10

        return queries
