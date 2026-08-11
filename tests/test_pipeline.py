"""Tests covering the parts that were silently wrong in the original project.

The leakage test is the important one: it is the check that would have caught
the inflated scores before they were written up as results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import N_BASE_IMAGES, assign_group_ids, group_split, verify_grouping
from src.evaluate import Predictions, bootstrap_ci, evaluate, threshold_sweep
from src.models import build_model, count_parameters


def _synthetic_index(n_base: int = 120, defect_fraction: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """Mimic the real CSV layout: n_base originals, then 3 variants each."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_base) < defect_fraction).astype(int)

    rows = [{"file": f"base_{i}.jpg", "label": int(labels[i])} for i in range(n_base)]
    for i in range(n_base):
        for v in range(3):
            rows.append({"file": f"aug_{i}_{v}.jpg", "label": int(labels[i])})

    df = pd.DataFrame(rows)
    df["group_id"] = assign_group_ids(len(df), n_base=n_base)
    df["is_original"] = np.arange(len(df)) < n_base
    df["defect_type"] = np.where(df["label"] == 1, "Defect_Shape", "Defect_No")
    return df


class TestGrouping:
    def test_group_ids_pair_originals_with_variants(self):
        groups = assign_group_ids(4900, n_base=N_BASE_IMAGES)
        assert len(groups) == 4900
        assert groups.max() == N_BASE_IMAGES - 1
        # every group must have exactly 4 members
        counts = np.bincount(groups)
        assert set(counts.tolist()) == {4}

    def test_first_group_matches_original_loader_offsets(self):
        # Original loop: group 1 = row 1, then rows 1226, 1227, 1228 (1-indexed).
        groups = assign_group_ids(4900, n_base=N_BASE_IMAGES)
        members = np.where(groups == 0)[0]  # 0-indexed
        assert members.tolist() == [0, 1225, 1226, 1227]

    def test_rejects_layout_that_breaks_assumption(self):
        with pytest.raises(ValueError):
            assign_group_ids(100, n_base=N_BASE_IMAGES)

    def test_verify_grouping_catches_mixed_labels(self):
        df = _synthetic_index(n_base=40)
        df.loc[len(df) - 1, "label"] = 1 - df.loc[len(df) - 1, "label"]
        with pytest.raises(ValueError, match="more than one label"):
            verify_grouping(df)


class TestSplitIsLeakageFree:
    def test_no_group_spans_two_splits(self):
        df = _synthetic_index(n_base=200)
        splits = group_split(df, seed=1)

        seen: dict[int, str] = {}
        for name, part in splits.items():
            for g in part["group_id"].unique():
                assert g not in seen, f"group {g} in both {seen.get(g)} and {name}"
                seen[g] = name

    def test_no_filename_appears_twice(self):
        df = _synthetic_index(n_base=200)
        splits = group_split(df, seed=1)
        files = pd.concat([p["file"] for p in splits.values()])
        assert files.duplicated().sum() == 0
        assert len(files) == len(df)

    def test_class_balance_preserved(self):
        df = _synthetic_index(n_base=400, defect_fraction=0.6)
        splits = group_split(df, seed=3)
        overall = df["label"].mean()
        for name, part in splits.items():
            assert abs(part["label"].mean() - overall) < 0.05, name

    def test_split_is_deterministic_given_seed(self):
        df = _synthetic_index(n_base=150)
        a = group_split(df, seed=7)["test"]["file"].tolist()
        b = group_split(df, seed=7)["test"]["file"].tolist()
        assert a == b

    def test_different_seeds_give_different_splits(self):
        df = _synthetic_index(n_base=150)
        a = set(group_split(df, seed=1)["test"]["file"])
        b = set(group_split(df, seed=2)["test"]["file"])
        assert a != b

    def test_ordered_split_would_have_leaked(self):
        """Demonstrates the original bug: sequential fill splits groups apart."""
        df = _synthetic_index(n_base=200)
        n = len(df)
        ordered = df.copy()
        ordered["split"] = [
            "train" if i < int(n * 0.6) else "valid" if i < int(n * 0.8) else "test"
            for i in range(n)
        ]
        spans = ordered.groupby("group_id")["split"].nunique()
        assert (spans > 1).sum() > 0, "ordered split should split groups across sets"


class TestModels:
    @pytest.mark.parametrize("name", ["cnn", "resnet", "se_resnet"])
    @pytest.mark.parametrize("width", [8, 64])
    def test_forward_shape(self, name, width):
        model = build_model(name, width=width)
        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 2)

    def test_width_reproduces_original_parameter_counts(self):
        # Counts recorded in the two original notebooks.
        assert count_parameters(build_model("cnn", width=8)) == 808_946
        assert count_parameters(build_model("cnn", width=64)) == 51_751_810

    def test_se_adds_few_parameters_over_plain_resnet(self):
        plain = count_parameters(build_model("resnet", width=64))
        se = count_parameters(build_model("se_resnet", width=64))
        assert se > plain
        assert (se - plain) / plain < 0.02  # SE is under 2% overhead

    def test_se_bottleneck_never_collapses_to_zero(self):
        # width=8 with reduction=16 would floor to 0 channels without the clamp.
        model = build_model("se_resnet", width=8, reduction=16)
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 2)

    def test_backward_pass_produces_gradients(self):
        model = build_model("se_resnet", width=8)
        loss = torch.nn.functional.cross_entropy(
            model(torch.randn(2, 3, 224, 224)), torch.tensor([0, 1])
        )
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestMetrics:
    def test_defect_recall_counts_missed_defects(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        probs = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 2 of 4 defects missed
        report = evaluate(Predictions(labels, probs))
        assert report.defect_recall == 0.5
        assert report.missed_defects == 2
        assert report.false_alarms == 0

    def test_weighted_average_hides_what_per_class_reveals(self):
        """The original metric choice, side by side with the honest one."""
        from sklearn.metrics import precision_recall_fscore_support

        # 90 ok, 10 defects; every defect missed.
        labels = np.array([0] * 90 + [1] * 10)
        probs = np.zeros(100)

        _, _, weighted_f1, _ = precision_recall_fscore_support(
            labels, (probs >= 0.5).astype(int), average="weighted", zero_division=0
        )
        report = evaluate(Predictions(labels, probs))

        assert weighted_f1 > 0.85          # looks excellent
        assert report.defect_recall == 0.0  # catches nothing at all

    def test_threshold_sweep_finds_recall_floor(self):
        rng = np.random.default_rng(0)
        labels = np.array([0] * 200 + [1] * 200)
        probs = np.concatenate([rng.beta(2, 5, 200), rng.beta(5, 2, 200)])
        result = threshold_sweep(Predictions(labels, probs), min_defect_recall=0.95)
        assert result["feasible"]
        assert result["defect_recall"] >= 0.95

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 400)
        probs = np.where(labels == 1, rng.uniform(0.5, 1, 400), rng.uniform(0, 0.5, 400))
        probs = np.clip(probs + rng.normal(0, 0.15, 400), 0, 1)

        point = evaluate(Predictions(labels, probs)).defect_recall
        lo, hi = bootstrap_ci(Predictions(labels, probs), n_boot=300, seed=0)
        assert lo <= point <= hi

    def test_ci_width_shows_small_test_sets_are_noisy(self):
        """~980 samples at 99% recall: the interval spans several images."""
        rng = np.random.default_rng(2)
        n = 980
        labels = np.array([1] * 600 + [0] * 380)
        probs = np.where(labels == 1, 0.99, 0.01)
        flip = rng.choice(np.where(labels == 1)[0], 6, replace=False)
        probs[flip] = 0.01  # 6 missed defects

        lo, hi = bootstrap_ci(Predictions(labels, probs), n_boot=500, seed=0)
        assert (hi - lo) > 0.005  # wider than the gaps the original ranked models on
