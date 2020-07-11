import math

import numpy as np
import pandas as pd

from src.configs.variables_const import VariablesConsts
from src.evaluation.ndcg_calculation import NDCG


class Metrics:
    """
    Class responsible to aggregate all metrics implemented
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        # self.ndcg = self._get_ndcg()
        # self.ap = self._get_average_precision()
        # self.rr = self._get_mean_reciprocal_rank()

        predicted = self.data.dropna(subset=[VariablesConsts.DISTANCE]).sort_values(by=VariablesConsts.DISTANCE, ascending=False)[
            VariablesConsts.PRODUCT_ID].values
        actual = self.data.sort_values(by=VariablesConsts.RELEVANCE, ascending=False)[VariablesConsts.PRODUCT_ID].values

        self.precision, self.recall = self.precision_and_recall(ranked_list=predicted, ground_truth=actual)
        self.ap = self.AP(ranked_list=predicted, ground_truth=actual)
        self.rr = self.RR(ranked_list=predicted, ground_truth=actual)
        self.ndcg = self.nDCG(ranked_list=predicted, ground_truth=actual)

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
        pred = self.data.dropna().sort_values(by=VariablesConsts.DISTANCE, ascending=False)[
            VariablesConsts.RELEVANCE].values
        pred = [1 if x == 3 else 0 for x in pred]
        rs = (np.asarray(r).nonzero()[0] for r in pred)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

    def _get_ndcg(self) -> float:
        pred_sorted = self.data.dropna().sort_values(by=VariablesConsts.DISTANCE, ascending=False)[
            VariablesConsts.RELEVANCE].values
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
        predicted = self.data.dropna()[VariablesConsts.PRODUCT_ID].values
        actual = self.data.sort_values(by=VariablesConsts.RELEVANCE, ascending=False)[VariablesConsts.PRODUCT_ID].values

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / len(actual)

    def nDCG(self, ranked_list, ground_truth):
        dcg = 0
        idcg = self.IDCG(len(ground_truth))
        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id not in ground_truth:
                continue
            rank = i + 1
            dcg += 1 / math.log(rank + 1, 2)
        return dcg / idcg

    def IDCG(self, n):
        idcg = 0
        for i in range(n):
            idcg += 1 / math.log(i + 2, 2)
        return idcg

    def AP(self, ranked_list, ground_truth):
        hits, sum_precs = 0, 0.0
        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id in ground_truth:
                hits += 1
                sum_precs += hits / (i + 1.0)
        if hits > 0:
            return sum_precs / len(ground_truth)
        else:
            return 0.0

    def RR(self, ranked_list, ground_truth):

        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id in ground_truth:
                return 1 / (i + 1.0)
        return 0

    def precision_and_recall(self, ranked_list, ground_truth):
        hits = 0
        for i in range(len(ranked_list)):
            id = ranked_list[i]
            if id in ground_truth:
                hits += 1

        if len(ranked_list):
            pre = hits / (1.0 * len(ranked_list))
        else:
            pre = 0
        rec = hits / (1.0 * len(ground_truth))
        return pre, rec
