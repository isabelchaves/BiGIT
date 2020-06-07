from typing import Tuple

import numpy as np
import pandas as pd
import ml_metrics as metrics
from scipy.spatial import distance
from sklearn.metrics import average_precision_score

from evaluation.ndcg_calculation import NDCG


class Metrics:
    """
    Class responsible to aggregate all metrics implemented
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.ndcg = self._get_ndcg()
        self.ap = self._get_average_precision()
        # self.mrr = self._get_mean_reciprocal_rank()

    # def _get_mean_reciprocal_rank(self, X, Y, indices, metric='hamming') -> Tuple[float, float]:
    #     """
    #     Original source: https://gist.github.com/craffel/449825abf1b46c6b8986
    #
    #     Computes the mean reciprocal rank of the correct match
    #     Assumes that X[n] should be closest to Y[n]
    #     Default uses hamming distance
    #     Parameters:
    #         - X : np.ndarray, shape=(n_examples, n_features)
    #             Data matrix in X modality
    #         - Y : np.ndarray, shape=(n_examples, n_features)
    #             Data matrix in Y modality
    #         - indices : np.ndarray
    #             Denotes which rows to use in MRR calculation
    #         - metric : str
    #             Which metric to use to compare feature vectors
    #     Returns:
    #         - mrr_pessimist : float
    #             Mean reciprocal rank, where ties are resolved pessimistically
    #             That is, rank = # of distances <= dist(X[:, n], Y[:, n])
    #         - mrr_optimist : float
    #             Mean reciprocal rank, where ties are resolved optimistically
    #             That is, rank = # of distances < dist(X[:, n], Y[:, n]) + 1
    #     """
    #     # Compute distances between each codeword and each other codeword
    #     distance_matrix = distance.cdist(X, Y, metric=metric)
    #     # Rank is the number of distances smaller than the correct distance, as
    #     # specified by the indices arg
    #     n_le = distance_matrix.T <= distance_matrix[np.arange(X.shape[0]), indices]
    #     n_lt = distance_matrix.T < distance_matrix[np.arange(X.shape[0]), indices]
    #     return (float(np.mean(1. / n_le.sum(axis=0))),
    #             float(np.mean(1. / (n_lt.sum(axis=0) + 1))))

    def _get_ndcg(self) -> float:
        pred_sorted = self.data.sort_values(by='distance')['relevance'].values
        ndcg_score = NDCG().ndcg_at_k(r=pred_sorted,
                                      k=pred_sorted.shape[0])
        return ndcg_score

    def _get_average_precision(self) -> float:
        pred_sorted = self.data.sort_values(by='distance')['distance'].values
        true_sorted = self.data.sort_values(by='relevance')['relevance'].values

        ap_score = average_precision_score(pred_sorted, true_sorted)

        return ap_score
