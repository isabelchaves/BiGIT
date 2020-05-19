import pandas as pd
from tqdm import tqdm

from evaluation.metrics import Metrics


class EvaluationMethod:
    def __init__(self, product_ids: dict):
        self.product_ids = product_ids

    # TODO: Improve performance here
    def _calculate_distances(self, data_dict: dict, vector_space_to_search, evaluate_column: str):
        distances_df = pd.DataFrame(columns=[evaluate_column, 'product_uid', 'distance'])

        for value in tqdm(data_dict, desc='Evaluate the products of the queries'):
            ids, distances = vector_space_to_search.knnQuery(data_dict[value], k=20)
            # print(query)
            distances_df = distances_df.append(
                pd.DataFrame([[value] + [x[1]] + [self.product_ids[x[0]]] for x in zip(ids, distances)],
                             columns=[evaluate_column, 'distance', 'product_uid']), ignore_index=True)

        return distances_df

    def run(self, data: pd.DataFrame, data_to_evaluate: dict, vector_space_to_search, evaluate_column: str):

        distances = self._calculate_distances(data_dict=data_to_evaluate,
                                              vector_space_to_search=vector_space_to_search,
                                              evaluate_column=evaluate_column)

        evaluate = data.merge(distances, on=['product_uid', evaluate_column], how='left')

        overall_ndcg = 0
        overall_map = 0
        overall_mrr = 0

        for value in list(evaluate[evaluate_column].unique()):
            products_to_evaluate = evaluate[evaluate[evaluate_column] == value]
            metrics = Metrics(data=products_to_evaluate)
            overall_ndcg += metrics.ndcg
            overall_map = metrics.map
            # overall_mrr = metrics.mrr

        print('Overall NDCG is {:.2f}'.format(overall_ndcg / evaluate[evaluate_column].nunique()))
        print('Overall MAP is {:.2f}'.format(overall_map / evaluate[evaluate_column].nunique()))
        # print('Overall MRR is {:.2f}'.format(overall_mrr / evaluate[evaluate_column].nunique()))
