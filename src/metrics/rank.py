import rank_eval

from typing import Dict, List, Optional, Union


class RankMetrics:
    def __init__(self):
        self.metrics = ["map", "mrr", "ndcg"]

    def compute_metrics(
        self,
        ground_truth,
        predictions,
        top_k: int = 1,
        metrics: Optional[List[str]] = None,
    ) -> Union[Dict[str, float], float]:
        metrics = self._validate_metrics(metrics) if metrics else self.metrics
        qrels = self.parse_ground_truth(ground_truth=ground_truth)
        run = self.parse_predictions(predictions=predictions)
        metrics = [f"{metric}@{top_k}" for metric in metrics]
        return rank_eval.evaluate(qrels, run, metrics)

    def _validate_metrics(self, metrics: List[str]) -> List[str]:
        for metric in metrics:
            if metric not in self.metrics:
                raise ValueError(
                    f"The metric `{metric}` is not supported. Currently, the following"
                    f" metrics are supported: {self.metrics}"
                )
        return metrics

    def parse_ground_truth(self, ground_truth) -> rank_eval.Qrels:
        qrels = rank_eval.Qrels()
        for query_id, docs_and_relevance in ground_truth.items():
            doc_ids, scores = zip(*docs_and_relevance)
            qrels.add(q_id=query_id, doc_ids=doc_ids, scores=scores)
        return qrels

    def parse_predictions(self, predictions) -> rank_eval.Run:
        run = rank_eval.Run()
        for query_id, docs_and_scores in predictions.items():
            doc_ids, scores = zip(*docs_and_scores)
            run.add(q_id=query_id, doc_ids=doc_ids, scores=scores)
        return run
