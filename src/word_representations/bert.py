from time import time

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from src.configs.variables_const import VariablesConsts
from src.data.processing.build_vector_spaces import build_vector_space
from src.evaluation.evaluation_method import EvaluationMethod
from src.graph_model.BiGIT import BiGIT


class BertExperiments:
    """
    http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    https://towardsdatascience.com/working-with-hugging-face-transformers-and-tf-2-0-89bf35e3555a
    https://huggingface.co/transformers/model_doc/bert.html
    """

    def __init__(self, word_vectors_strategy: str, vector_method: str, vector_space: str, product_ids: dict):
        self.word_vectors_strategy = word_vectors_strategy
        self.vector_method = vector_method
        self.vector_space = vector_space
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertModel.from_pretrained('bert-base-uncased')
        self.evaluation = EvaluationMethod(product_ids=product_ids)

    def prepare_data(self, data):
        start_time = time()

        product_title_vectors = data[VariablesConsts.PRODUCT_TITLE_PROCESSED].apply(
            lambda x: self.model(tf.constant(self.tokenizer.encode(x))[None, :])[0][:, 0, :].numpy())

        self.vector_size = len(product_title_vectors[0])

        search_term_vectors = data[VariablesConsts.SEARCH_TERM_PROCESSED].apply(
            lambda x: self.model(tf.constant(self.tokenizer.encode(x))[None, :])[0][:, 0, :].numpy())

        print(time() - start_time)

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
        print('##########################    BiGIT    ############################')
        print('###################################################################')

        products, queries = self.prepare_data(data=data)

        bigit = BiGIT(dimensions=self.vector_size,
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
