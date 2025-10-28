import numpy as np


class Metrics:
    def __init__(self, metric_list, k_list=(1, 5, 10)):
        """
        Initialize retrieval metrics.

        Args:
            metric_list (tuple): Metrics to compute ("precision", "recall", "ndcg").
            k_list (tuple): List of K values (e.g., (1, 5, 10)) for evaluation.
        """
        self.metric_list = metric_list
        self.k_list = sorted(k_list)  # Ensure K is in ascending order

    def precision_at_k(self, prediction, label, k):
        """
        Compute Precision@K.
        """
        return 1.0 if label in prediction[:k] else 0.0  # Precision is either 1 or 0

    def recall_at_k(self, prediction, label, k):
        """
        Compute Recall@K.
        """
        return 1.0 if label in prediction[:k] else 0.0  # Recall is 1 if label is in top-k

    def multi_label_hit_style_recall(self, prediction, label, k):
        """
        Compute Multi-label Recall@k, in hit style.
        """
        return 1.0 if len(set(prediction[:k]).intersection(set(label))) > 0 else 0.0  # Recall is 1 if label is in top-k

    def ndcg_at_k(self, prediction, label, k):
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@K).
        """
        def dcg(rel_list):
            return sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(rel_list))

        relevance = [1 if item == label else 0 for item in prediction[:k]]
        ideal_relevance = sorted(relevance, reverse=True)

        dcg_score = dcg(relevance)
        idcg_score = dcg(ideal_relevance) if any(ideal_relevance) else 1  # Avoid division by zero

        return dcg_score / idcg_score if idcg_score > 0 else 0.0

    def evaluate(self, test_cases):
        metric_results = {f"{metric}@{k}": [] for metric in self.metric_list for k in self.k_list}

        for case in test_cases:
            prediction, label = case["prediction"], case["label"]  # Label is a string

            for k in self.k_list:
                if "precision" in self.metric_list:
                    metric_results[f"precision@{k}"].append(self.precision_at_k(prediction, label, k))
                if "recall" in self.metric_list:
                    metric_results[f"recall@{k}"].append(self.recall_at_k(prediction, label, k))
                if "ndcg" in self.metric_list:
                    metric_results[f"ndcg@{k}"].append(self.ndcg_at_k(prediction, label, k))
                if "multi-label recall" in self.metric_list:
                    metric_results[f"multi-label recall@{k}"].append(self.multi_label_hit_style_recall(prediction, label, k))

        return {key: float(np.mean(values)) for key, values in metric_results.items()}
