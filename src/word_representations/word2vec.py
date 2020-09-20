import gensim.downloader as api
import numpy as np
import pandas as pd

from src.configs.variables_const import VariablesConsts
from src.data.processing.build_vector_spaces import build_vector_space
from src.evaluation.evaluation_method import EvaluationMethod
from src.graph_model.BiGIT import BiGIT


class Word2VecExperiments:
    def __init__(self, word_vectors_strategy: str, vector_method: str, vector_space: str, product_ids: dict):
        self.word_vectors_strategy = word_vectors_strategy
        self.vector_method = vector_method
        self.vector_space = vector_space
        self.model = api.load('word2vec-google-news-300')
        self.evaluation = EvaluationMethod(product_ids=product_ids)

    def _calculate_word_vectors(self, word_vectors, list_of_words):
        """
        https://stackoverflow.com/questions/46889727/word2vec-what-is-best-add-concatenate-or-average-word-vectors
        """
        vectors = []

        for word in list_of_words:
            if len(word) > 0:
                try:
                    vector = word_vectors[word]
                    vector.astype(np.float16)
                    vectors.append(vector)

                except KeyError:
                    nans_vector = np.zeros(300, dtype=int) + np.nan
                    vectors.append(nans_vector)
            pass

        if self.word_vectors_strategy == 'sum':
            return np.nansum([vectors], axis=1).flatten()

        elif self.word_vectors_strategy == 'average':
            return np.nanmean([vectors], axis=1).flatten()

        else:
            # return np.array([vectors])
            raise ValueError('strategy must be either \'sum\' or \'average\'')

    def prepare_data(self, data):

        product_title_vectors = data[VariablesConsts.PRODUCT_TITLE_PROCESSED].apply(
            lambda x: self._calculate_word_vectors(word_vectors=self.model,
                                                   list_of_words=x))
        search_term_vectors = data[VariablesConsts.SEARCH_TERM_PROCESSED].apply(
            lambda x: self._calculate_word_vectors(word_vectors=self.model,
                                                   list_of_words=x))

        products = dict()
        queries = dict()

        for i, row in data.iterrows():
            if row[VariablesConsts.PRODUCT_ID] not in products:
                products[row[VariablesConsts.PRODUCT_ID]] = product_title_vectors[i]

            if row[VariablesConsts.SEARCH_TERM_PROCESSED] not in queries:
                queries[row[VariablesConsts.SEARCH_TERM_PROCESSED]] = search_term_vectors[i]

        return products, queries

    def run_baseline(self, data: pd.DataFrame):
        print('###################################################################')
        print('########################   BASELINE   #############################')
        print('###################################################################')

        products, queries = self.prepare_data(data=data)

        # Building Products vector space
        product_vs = build_vector_space(data=products, vector_method=self.vector_method, vector_space=self.vector_space)

        # Building Query vector space
        query_vs = build_vector_space(data=queries, vector_method=self.vector_method, vector_space=self.vector_space)

        self.evaluation.run(data=data,
                            data_to_evaluate=queries,
                            vector_space_to_search=product_vs,
                            evaluate_column=VariablesConsts.SEARCH_TERM_PROCESSED)

        return product_vs, query_vs

    def run_with_bigit(self, data: pd.DataFrame, bigit_interaction_number: int, bigit_initialization: str):

        print('###################################################################')
        print('########################   CLICK GRAPH   ##########################')
        print('###################################################################')

        products, queries = self.prepare_data(data=data)

        bigit = BiGIT(dimensions=self.model.vector_size,
                      data=data)

        queries, products = bigit.run(products=products,
                                      queries=queries,
                                      iterations_nr=bigit_interaction_number,
                                      start=bigit_initialization)

        # Building Products vector space
        product_vs = build_vector_space(data=products, vector_method=self.vector_method, vector_space=self.vector_space)

        # Building Query vector space
        query_vs = build_vector_space(data=queries, vector_method=self.vector_method, vector_space=self.vector_space)

        print('QUERIES ANALYSIS')

        self.evaluation.run(data=data,
                            data_to_evaluate=queries,
                            vector_space_to_search=product_vs,
                            evaluate_column=VariablesConsts.SEARCH_TERM_PROCESSED)

        # print('PRODUCTS ANALYSIS')
        #
        # EvaluationMethod().run(data=data,
        #                        data_to_evaluate=products,
        #                        vector_space_to_search=query_vs,
        #                        evaluate_column='product_title_processed')

        return product_vs, query_vs
