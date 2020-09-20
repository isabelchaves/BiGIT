import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.configs.variables_const import VariablesConsts


class BiGIT:
    def __init__(self, dimensions: int, data: pd.DataFrame):
        self.dimensions = dimensions
        self.G = self._create_graph(data=data)

    def _create_graph(self, data: pd.DataFrame):
        G = nx.Graph()
        G.add_nodes_from(data[VariablesConsts.PRODUCT_ID].unique(), bipartite=0)
        G.add_nodes_from(data.search_term_processed.unique(), bipartite=1)
        G.add_weighted_edges_from(
            list(data[[VariablesConsts.PRODUCT_ID, VariablesConsts.SEARCH_TERM_PROCESSED,
                       VariablesConsts.CLICK_SCORE]].to_records(index=False)))
        return G

    def __transfer_queries_to_products(self, products: np.array, queries: np.array):
        # Products <- Queries
        for A in tqdm(products, desc='Products <- Queries'):
            update = np.zeros(self.dimensions)
            total_weight = 0
            for B, value in self.G[A].items():
                update = update + value['weight'] * queries[B]
                total_weight += value['weight']
            if total_weight != 0:
                products[A] += update / total_weight
                products[A] /= 2

        return products, queries

    def __transfer_products_to_queries(self, products: np.array, queries: np.array):
        # Queries <- Products
        for B in tqdm(queries, desc='Queries <- Products'):
            update = np.zeros(self.dimensions)
            total_weight = 0
            for A, value in self.G[B].items():
                update = update + value['weight'] * products[A]
                total_weight += value['weight']
            if total_weight != 0:
                queries[B] += update / total_weight
                queries[B] /= 2

        return products, queries

    def run(self, products: np.array, queries: np.array, iterations_nr: int, start: str):
        if start == 'document':
            for iter in range(iterations_nr):
                # Products <- Queries
                products, queries = self.__transfer_queries_to_products(products=products, queries=queries)
                # Queries <- Products
                products, queries = self.__transfer_products_to_queries(products=products, queries=queries)

        elif start == 'query':
            for iter in range(iterations_nr):
                # Queries <- Products
                products, queries = self.__transfer_products_to_queries(products=products, queries=queries)
                # Products <- Queries
                products, queries = self.__transfer_queries_to_products(products=products, queries=queries)

        else:
            raise Exception('Method name {} not implemented'.format(start))

        return queries, products
