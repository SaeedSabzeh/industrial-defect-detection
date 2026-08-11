"""Evaluation focused on the metric a production line actually cares about.

The original project reported
`precision_recall_fscore_support(..., average="weighted")`, which averages
across classes in proportion to their support. On a defect-detection task
that hides the only number that matters: how many defective units slipped
through. A missed defect ships a bad product to a customer; a false alarm
discards a good one. Those costs are not symmetric and a weighted average
treats them as if they were.

Everything here treats **defect (nok) as the positive class** and reports
defect recall separately, with bootstrap confidence intervals so that
differences between models can be judged against sampling noise rather than
eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

__all__ = ["Predictions", "collect_predictions", "EvalReport", "evaluate", "bootstrap_ci", "threshold_sweep", "format_report"]

DEFECT = 1  # positive class
OK = 0


@dataclass
class Predictions:
    labels: np.ndarray        # (n,) ground truth, 1 = defect
    defect_prob: np.ndarray   # (n,) predicted probability of defect


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: torch.device | None = None) -> Predictions:
    """Run the model once and keep probabilities, not just argmax labels.

    Storing probabilities is what makes threshold selection possible later --
    argmax hard-codes a 0.5 operating point that is almost never the right
    one for asymmetric costs.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_labels, all_probs = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = F.softmax(logits, dim=1)[:, DEFECT]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return Predictions(np.concatenate(all_labels), np.concatenate(all_probs))


@dataclass
class EvalReport:
    threshold: float
    n_samples: int
    n_defects: int
    defect_recall: float          # fraction of defects caught
    defect_precision: float
    defect_f1: float
    ok_recall: float              # 1 - false alarm rate
    false_alarm_rate: float
    missed_defects: int
    false_alarms: int
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    average_precision: float
    confusion: list[list[int]]    # [[TN, FP], [FN, TP]]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(preds: Predictions, threshold: float = 0.5) -> EvalReport:
    """Full report at a given operating threshold."""
    y_true = preds.labels
    y_pred = (preds.defect_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[OK, DEFECT])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[OK, DEFECT], zero_division=0
    )

    n_defects = int((y_true == DEFECT).sum())
    n_ok = int((y_true == OK).sum())

    # AUC is undefined if only one class is present.
    if n_defects and n_ok:
        auc = float(roc_auc_score(y_true, preds.defect_prob))
        ap = float(average_precision_score(y_true, preds.defect_prob))
    else:
        auc = ap = float("nan")

    defect_recall = float(recall[1])
    ok_recall = float(recall[0])

    return EvalReport(
        threshold=threshold,
        n_samples=len(y_true),
        n_defects=n_defects,
        defect_recall=defect_recall,
        defect_precision=float(precision[1]),
        defect_f1=float(f1[1]),
        ok_recall=ok_recall,
        false_alarm_rate=1.0 - ok_recall,
        missed_defects=int(fn),
        false_alarms=int(fp),
        accuracy=float((y_true == y_pred).mean()),
        balanced_accuracy=float((defect_recall + ok_recall) / 2),
        roc_auc=auc,
        average_precision=ap,
        confusion=cm.tolist(),
    )


def bootstrap_ci(
    preds: Predictions,
    metric: str = "defect_recall",
    threshold: float = 0.5,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric.

    On a ~980-image test set at 99% accuracy, the gap between two models is
    often fewer than ten images. Without an interval there is no way to tell
    a real improvement from resampling noise, and the original project's
    model ranking rests entirely on differences of that size.
    """
    rng = np.random.default_rng(seed)
    n = len(preds.labels)
    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = Predictions(preds.labels[idx], preds.defect_prob[idx])
        if len(np.unique(sample.labels)) < 2:
            continue
        values.append(getattr(evaluate(sample, threshold=threshold), metric))

    if not values:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def threshold_sweep(preds: Predictions, min_defect_recall: float = 0.99) -> dict:
    """Find the threshold meeting a defect-recall floor at least false alarms.

    This is how the operating point should be chosen on a real line: fix the
    tolerable escape rate first, then minimise the good units thrown away.
    """
    precision, recall, thresholds = precision_recall_curve(preds.labels, preds.defect_prob)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    feasible = [
        (t, p, r)
        for t, p, r in zip(thresholds, precision[:-1], recall[:-1])
        if r >= min_defect_recall
    ]
    if not feasible:
        return {
            "feasible": False,
            "min_defect_recall": min_defect_recall,
            "best_achievable_recall": float(recall.max()),
        }

    # Among thresholds meeting the recall floor, take the highest precision.
    t, p, r = max(feasible, key=lambda x: x[1])
    return {
        "feasible": True,
        "min_defect_recall": min_defect_recall,
        "threshold": float(t),
        "defect_precision": float(p),
        "defect_recall": float(r),
    }


def format_report(report: EvalReport, name: str = "model", ci: tuple[float, float] | None = None) -> str:
    tn, fp = report.confusion[0]
    fn, tp = report.confusion[1]
    ci_text = f"  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""

    return "\n".join(
        [
            f"=== {name} @ threshold {report.threshold:.2f} ===",
            f"  samples {report.n_samples}  (defects {report.n_defects})",
            f"  defect recall     {report.defect_recall:.4f}{ci_text}",
            f"  defect precision  {report.defect_precision:.4f}",
            f"  defect F1         {report.defect_f1:.4f}",
            f"  false alarm rate  {report.false_alarm_rate:.4f}",
            f"  missed defects    {report.missed_defects}",
            f"  false alarms      {report.false_alarms}",
            f"  balanced accuracy {report.balanced_accuracy:.4f}",
            f"  ROC AUC           {report.roc_auc:.4f}",
            f"  avg precision     {report.average_precision:.4f}",
            f"  confusion  [[TN {tn}, FP {fp}], [FN {fn}, TP {tp}]]",
        ]
    )
