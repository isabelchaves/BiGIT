import pandas as pd
from tqdm import tqdm

from src.configs.variables_const import VariablesConsts
from src.evaluation.metrics import Metrics


class EvaluationMethod:
    def __init__(self, product_ids: dict):
        self.product_ids = product_ids

    # TODO: Improve performance here
    def _calculate_distances(self, data_dict: dict, vector_space_to_search, evaluate_column: str):
        distances_df = pd.DataFrame(columns=[evaluate_column, VariablesConsts.PRODUCT_ID, VariablesConsts.DISTANCE])

        for value in tqdm(data_dict, desc='Evaluate the products of the queries'):
            ids, distance = vector_space_to_search.knnQuery(data_dict[value], k=20)
            # print(query)
            distances_df = distances_df.append(
                pd.DataFrame([[value] + [x[1]] + [self.product_ids[x[0]]] for x in zip(ids, distance)],
                             columns=[evaluate_column, VariablesConsts.DISTANCE, VariablesConsts.PRODUCT_ID]),
                ignore_index=True)

        return distances_df

    def run(self, data: pd.DataFrame, data_to_evaluate: dict, vector_space_to_search, evaluate_column: str):

        distances = self._calculate_distances(data_dict=data_to_evaluate,
                                              vector_space_to_search=vector_space_to_search,
                                              evaluate_column=evaluate_column)

        evaluate = data.merge(distances, on=[VariablesConsts.PRODUCT_ID, evaluate_column], how='left')

        precision_list = []
        recall_list = []
        ap_list = []
        ndcg_list = []
        rr_list = []

        for value in list(evaluate[evaluate_column].unique()):
            products_to_evaluate = evaluate[evaluate[evaluate_column] == value]
            metrics = Metrics(data=products_to_evaluate)
            precision_list.append(metrics.precision)
            recall_list.append(metrics.recall)
            ap_list.append(metrics.ap)
            rr_list.append(metrics.rr)
            ndcg_list.append(metrics.ndcg)

        precison = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = 2 * precison * recall / (precison + recall)
        map = sum(ap_list) / len(ap_list)
        mrr = sum(rr_list) / len(rr_list)
        mndcg = sum(ndcg_list) / len(ndcg_list)

        print('Metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f' % (
            round(f1, 4), round(map, 4), round(mrr, 4), round(mndcg, 4)))
