import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler

from evaluation.ndcg_calculation import NDCG


class Metrics:
    """
    Class responsible to aggregate all metrics implemented
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.ndcg = self._get_ndcg()
        self.ap = self._get_average_precision()
        self.mrr = self._get_mean_reciprocal_rank()

    def _get_mean_reciprocal_rank(self):
        """
        Adapted from: https://gist.github.com/bwhite/3726239
        Score is reciprocal of the rank of the first relevant item
        First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
        Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
        >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        >>> mean_reciprocal_rank(rs)
        0.61111111111111105
        >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        >>> mean_reciprocal_rank(rs)
        0.5
        >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
        >>> mean_reciprocal_rank(rs)
        0.75
        Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Mean reciprocal rank
        """
        pred = self.data.fillna(0).sort_values(by='relevance', ascending=False)['distance'].values
        pred = MinMaxScaler(feature_range=(0, 1)).fit_transform(pred.reshape(-1, 1)).reshape(1, -1)
        rs = (np.asarray(r).nonzero()[0] for r in pred)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    def _get_ndcg(self) -> float:
        pred_sorted = self.data.sort_values(by='distance', ascending=False)['relevance'].values
        ndcg_score = NDCG().ndcg_at_k(r=pred_sorted,
                                      k=pred_sorted.shape[0])
        return ndcg_score

    def _get_average_precision(self) -> float:
        pred_sorted = self.data.fillna(0).sort_values(by='distance', ascending=False)['distance'].values
        true_sorted = self.data.sort_values(by='relevance', ascending=False)['relevance'].values
        true_sorted = [1 if x > 2.5 else 0 for x in true_sorted]

        return average_precision_score(true_sorted, pred_sorted)
