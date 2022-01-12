from abc import ABCMeta, abstractmethod

import numpy as np


class Metrics(metaclass=ABCMeta):

    @abstractmethod
    def __str__(self):
        pass


class Accuracy(Metrics):

    def __str__(self):
        return "acc"

    def __call__(self, output: list, label: list) -> float:
        """get accuracy
        Args:
            output : model prediction
            label : label
        Returns:
            float: accuracy
        """
        total = len(output)
        label_array = np.array(label)
        output_array = np.array(output)

        assert len(label_array) == len(output_array)
        match = np.sum(label_array == output_array)
        return match / total


class nDCG(Metrics):

    def __str__(self):
        return "ndcg"

    def __call__(self, output: list, label: list) -> float:
        """get normalized Discounted Cumulative Gain

        Args:
            output : model prediction
            label : label
        Returns:
            float: ndcg
        """
        label = label[:len(output)]  # negative sampling 시 label 의 개수가 output 보다 많음
        hits = np.array(output) == np.array(label).reshape(-1, 1)
        k = np.array(output).shape[-1]
        dcg_weight = 1 / np.log2(np.arange(2, k + 2))

        idcg = 1.
        dcg = np.sum(hits.astype(float) * dcg_weight, axis=-1, keepdims=True)
        return np.mean(dcg / idcg)


class RecallAtK(Metrics):

    def __str__(self):
        return "recall@k"

    def __call__(self, output: list, label: list) -> float:
        """get recall at k
        Args:
            output : model prediction
            label : label
        Returns:
            float: recall@k
        """
        label = label[:len(output)]  # negative sampling 시 label 의 개수가 output 보다 많음
        n = len(label)
        hits = np.array(output) == np.array(label).reshape(-1, 1)
        n_hits = np.sum(hits)
        return n_hits / n


class PrecisionAtK(Metrics):

    def __str__(self):
        return "precision@k"

    def __call__(self, output: list, label: list) -> float:
        """get precision at k
        Args:
            output : model prediction
            label : label
        Returns:
            float: precision@k
        """
        k = np.array(output).shape[-1]
        hits = np.array(output) == np.array(label).reshape(-1, 1)
        precision = np.sum(hits, axis=-1) / k
        return np.mean(precision)

