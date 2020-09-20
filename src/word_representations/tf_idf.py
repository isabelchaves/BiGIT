import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.configs.variables_const import VariablesConsts
from src.data.processing.build_vector_spaces import build_vector_space
from src.evaluation.evaluation_method import EvaluationMethod
from src.graph_model.BiGIT import BiGIT


class TfIfdExperiments:
    def __init__(self, vector_space: str, vector_method: str, product_ids: dict):
        self.vector_space = vector_space
        self.vector_method = vector_method
        self.model = None
        self.product_ids = product_ids
        self.evaluation = EvaluationMethod(product_ids=product_ids)

    def _tf_idf_train(self, corpus):
        corpus = list(dict.fromkeys(corpus))
        model = TfidfVectorizer(stop_words='english').fit(corpus)
        return model

    def prepare_data(self, data):
        if not self.model:
            corpus = data[VariablesConsts.PRODUCT_TITLE_PROCESSED].values.tolist() + data[
                VariablesConsts.SEARCH_TERM_PROCESSED].values.tolist()
            self.model = self._tf_idf_train(corpus=corpus)

        product_title_vectors = self.model.transform(data[VariablesConsts.PRODUCT_TITLE_PROCESSED])
        search_term_vectors = self.model.transform(data[VariablesConsts.SEARCH_TERM_PROCESSED])

        products = dict()
        queries = dict()

        for i, row in data.iterrows():
            if row[VariablesConsts.PRODUCT_ID] not in products:
                products[row[VariablesConsts.PRODUCT_ID]] = product_title_vectors[i].todense()

            if row[VariablesConsts.SEARCH_TERM_PROCESSED] not in queries:
                queries[row[VariablesConsts.SEARCH_TERM_PROCESSED]] = search_term_vectors[i].todense()

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
        print('##########################    BiGIT    ############################')
        print('###################################################################')

        products, queries = self.prepare_data(data=data)

        bigit = BiGIT(dimensions=len(self.model.vocabulary_),
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
