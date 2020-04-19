import pandas as pd

from click_graph.click_graph_model import ClickGraphModel
from data.processing.build_vector_spaces import build_vector_space
from evaluation.evaluation_method import EvaluationMethod
from tf_idf.tf_idf_implementation import TfIdfImplementation


class TfIfdExperiments:
    def __init__(self, vector_space: str, vector_method: str):
        self.vector_space = vector_space
        self.vector_method = vector_method

    def run(self, data):
        corpus = data['product_title_processed'].values.tolist() + data['search_term_processed'].values.tolist()
        model = TfIdfImplementation(corpus=corpus).model
        product_title_vectors = model.transform(data['product_title_processed'])
        search_term_vectors = model.transform(data['search_term_processed'])

        products = dict()
        queries = dict()

        for i, row in data.iterrows():
            if row['product_uid'] not in products:
                products[row['product_uid']] = product_title_vectors[i].todense()

            if row['search_term_processed'] not in queries:
                queries[row['search_term_processed']] = search_term_vectors[i].todense()

        self.run_baseline(data=data,
                          products=products,
                          queries=queries)

        self.run_with_click_graph(data=data,
                                  products=products,
                                  queries=queries,
                                  dimensions=len(model.vocabulary_))

    def run_baseline(self, data: pd.DataFrame, products, queries):
        print('###################################################################')
        print('########################   BASELINE   #############################')
        print('###################################################################')

        # Building Products vector space
        product_vs = build_vector_space(data=products, vector_method=self.vector_method, vector_space=self.vector_space)

        # Building Query vector space
        # query_vs = build_vector_space(data=queries, vector_method=self.vector_method, vector_space=self.vector_space)

        EvaluationMethod().run(data=data,
                               data_to_evaluate=queries,
                               vector_space_to_search=product_vs,
                               evaluate_column='search_term_processed')

    def run_with_click_graph(self, data: pd.DataFrame, products, queries, dimensions):

        print('###################################################################')
        print('########################   CLICK GRAPH   ##########################')
        print('###################################################################')

        click_graph = ClickGraphModel(dimensions=dimensions,
                                      data=data)

        queries, products = click_graph.run(products=products,
                                            queries=queries,
                                            iterations_nr=10,
                                            start='document')

        # Building Products vector space
        product_vs = build_vector_space(data=products, vector_method=self.vector_method, vector_space=self.vector_space)

        # Building Query vector space
        # query_vs = build_vector_space(data=queries, vector_method=self.vector_method, vector_space=self.vector_space)

        print('QUERIES ANALYSIS')

        EvaluationMethod().run(data=data,
                               data_to_evaluate=queries,
                               vector_space_to_search=product_vs,
                               evaluate_column='search_term_processed')

        # print('PRODUCTS ANALYSIS')
        #
        # EvaluationMethod().run(data=data,
        #                        data_to_evaluate=products,
        #                        vector_space_to_search=query_vs,
        #                        evaluate_column='product_title_processed')
