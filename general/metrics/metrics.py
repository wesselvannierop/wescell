import torch
from torchmetrics.classification import (
    BinaryJaccardIndex,
    Dice,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
)
from torchmetrics.metric import Metric


class MetricsBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.jaccard = BinaryJaccardIndex()
        self.dice = Dice(num_classes=1, multiclass=False)
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        self.accuracy = BinaryAccuracy()

    def dict(self) -> dict:
        return {
            "jaccard": self.jaccard,
            "dice": self.dice,
            "recall": self.recall,
            "precision": self.precision,
            "accuracy": self.accuracy,
        }


class CellConfluencyMetrics(Metric):
    def __init__(self) -> None:
        """Computes common metrics such as Jaccard, Dice, Recall, Precision and Accuracy for a
        binary segmentation problem. It can deal with inverted predictions compared to the
        targets."""
        super().__init__()
        self.metrics = MetricsBase()
        self.metrics_inv = MetricsBase()

    def update(self, pred, gt):
        for metric, metric_inv in zip(
            self.metrics.dict().values(), self.metrics_inv.dict().values()
        ):
            metric.update(pred.to(torch.int), gt.to(torch.int))
            metric_inv.update(pred.logical_not().to(torch.int), gt.to(torch.int))

    def reset(self):
        for metric, metric_inv in zip(
            self.metrics.dict().values(), self.metrics_inv.dict().values()
        ):
            metric.reset()
            metric_inv.reset()

    def compute(self, return_match=False):
        report, report_inv = self.report()
        if report["dice"] > report_inv["dice"]:
            if return_match:
                return report, {0: 0, 1: 1}
            else:
                return report
        else:
            if return_match:
                return report_inv, {0: 1, 1: 0}
            else:
                return report_inv

    def report(self):
        report = {}
        report_inv = {}
        for (name, metric), metric_inv in zip(
            self.metrics.dict().items(), self.metrics_inv.dict().values()
        ):
            report[name] = metric.compute().item()
            report_inv[name] = metric_inv.compute().item()
        return report, report_inv

    @staticmethod
    def logger_format(report: dict, log_group: str = "Metrics"):
        """Format the generated report to be used with a PyTorch Lightning logger

        Args:
            report (dict): Dict of the metric scores.
            log_group (str, optional): Prepend a log group. Defaults to "Metrics".

        Returns:
            _type_: _description_
        """
        logs = []
        for key, value in report.items():
            logs.append(
                {
                    "name": log_group + "/" + key,
                    "value": value,
                }
            )
        return logs
