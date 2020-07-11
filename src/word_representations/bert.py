import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel

from src.click_graph.click_graph_model import ClickGraphModel
from src.configs.variables_const import VariablesConsts
from src.data.processing.build_vector_spaces import build_vector_space
from src.evaluation.evaluation_method import EvaluationMethod


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
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.model = TFBertModel.from_pretrained('bert-large-uncased')
        self.evaluation = EvaluationMethod(product_ids=product_ids)

    def tokenize(self, sentences):
        input_ids, input_masks, input_segments = [], [], []
        for sentence in tqdm(sentences):
            inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                                return_attention_mask=True, return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])

        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments,
                                                                                                        dtype='int32')

    def _tokenize(self, data: pd.Series):

        tokenized = data.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = tf.constant(padded)
        attention_mask = tf.constant(attention_mask)

        return input_ids, attention_mask

    def prepare_data(self, data):

        # input_ids, attention_mask = self._tokenize(data[VariablesConsts.PRODUCT_TITLE])
        product_title_vectors = data[VariablesConsts.PRODUCT_TITLE].apply(
            lambda x: self.model(tf.constant(self.tokenizer.encode(x))[None, :])[0][:, 0, :].numpy())
        # product_title_vectors = last_hidden_states[0][:, 0, :].numpy()

        search_term_vectors = data[VariablesConsts.SEARCH_TERM].apply(
            lambda x: self.model(tf.constant(self.tokenizer.encode(x))[None, :])[1])

        print('Done cenas')

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

    def run_with_click_graph(self, data: pd.DataFrame, click_graph_interaction_number: int):

        print('###################################################################')
        print('########################   CLICK GRAPH   ##########################')
        print('###################################################################')

        products, queries = self.prepare_data(data=data)

        click_graph = ClickGraphModel(dimensions=self.model.vector_size,
                                      data=data)

        queries, products = click_graph.run(products=products,
                                            queries=queries,
                                            iterations_nr=click_graph_interaction_number,
                                            start='document')

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
