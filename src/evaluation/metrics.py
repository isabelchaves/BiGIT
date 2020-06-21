import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.evaluation import NDCG


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
        pred = self.data.sort_values(by='distance', ascending=False)['relevance'].values
        pred = MinMaxScaler(feature_range=(0, 1)).fit_transform(pred.reshape(-1, 1)).reshape(1, -1)
        rs = (np.asarray(r).nonzero()[0] for r in pred)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    def _get_ndcg(self) -> float:
        pred_sorted = self.data.sort_values(by='distance', ascending=False)['relevance'].values
        ndcg_score = NDCG().ndcg_at_k(r=pred_sorted,
                                      k=pred_sorted.shape[0])
        return ndcg_score

    def _get_average_precision(self) -> float:

        """
        Adapted from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

        Computes the average precision at k.
        This function computes the average prescision at k between two lists of items.

        Variables needed:
        -----------------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        --------------
        score : double
                The average precision at k (size of actual list) over the input lists
        """
        predicted = self.data.sort_values(by='distance', ascending=False)['distance'].values
        actual = self.data.sort_values(by='relevance', ascending=False)['relevance'].values

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / len(actual)
