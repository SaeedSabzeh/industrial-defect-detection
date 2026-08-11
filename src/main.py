"""Command-line entry point.

Examples
--------
Train one model:
    python -m src.main train --model se_resnet --width 64 --epochs 20

Reproduce the width sweep that replaces the two original notebooks:
    python -m src.main sweep --widths 8 64 --seeds 0 1 2

Quantify how much the original split's leakage inflated the scores:
    python -m src.main leakage-check --model se_resnet --width 8
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch

from .data import build_index, group_split, make_loaders, verify_grouping
from .evaluate import bootstrap_ci, collect_predictions, evaluate, format_report, threshold_sweep
from .models import build_model, count_parameters
from .train import TrainConfig, set_seed, train_model


def _resolve_paths(args) -> tuple[str, str]:
    """Locate images dir and annotation CSV, downloading via kagglehub if needed."""
    if args.data_root:
        root = args.data_root
    else:
        import kagglehub  # imported lazily so offline runs work with --data-root

        path = kagglehub.dataset_download("imonbilk/industry-biscuit-cookie-dataset")
        root = os.path.join(path, "IndustryBiscuit")

    return os.path.join(root, "Images"), os.path.join(root, "Annotations.csv")


def _prepare(args):
    images_dir, csv_path = _resolve_paths(args)
    index = build_index(csv_path)
    print(verify_grouping(index).to_string(index=False))

    splits = group_split(index, seed=args.seed)
    print("\nsplit sizes (images):")
    for name in ("train", "valid", "test"):
        part = splits[name]
        n_defect = int(part["label"].sum())
        print(f"  {name:5s} {len(part):5d}  defects {n_defect:5d}  groups {part['group_id'].nunique():4d}")

    loaders = make_loaders(
        splits,
        images_dir,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        seed=args.seed,
        originals_only_train=args.originals_only_train,
    )
    return loaders


def cmd_train(args) -> None:
    set_seed(args.seed)
    loaders = _prepare(args)

    cfg = TrainConfig(
        model=args.model,
        width=args.width,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        augment=not args.no_augment,
        originals_only_train=args.originals_only_train,
        output_dir=args.output_dir,
    )
    model = build_model(cfg.model, width=cfg.width, reduction=cfg.reduction)
    print(f"\n{cfg.model} (width={cfg.width}): {count_parameters(model):,} trainable parameters\n")

    model, _ = train_model(model, loaders, cfg)

    preds = collect_predictions(model, loaders["test"])
    report = evaluate(preds)
    ci = bootstrap_ci(preds, "defect_recall")
    print()
    print(format_report(report, f"{cfg.model} w{cfg.width}", ci=ci))
    print()
    print("operating point for 99% defect recall:", json.dumps(threshold_sweep(preds), indent=2))


def cmd_sweep(args) -> None:
    """Train every (model, width, seed) combination and tabulate with CIs."""
    rows = []
    for width in args.widths:
        for name in args.models:
            for seed in args.seeds:
                set_seed(seed)
                args.seed = seed
                loaders = _prepare(args)

                cfg = TrainConfig(
                    model=name,
                    width=width,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    seed=seed,
                    augment=not args.no_augment,
                    output_dir=args.output_dir,
                )
                model = build_model(name, width=width)
                params = count_parameters(model)
                model, _ = train_model(model, loaders, cfg, verbose=not args.quiet)

                preds = collect_predictions(model, loaders["test"])
                report = evaluate(preds)
                rows.append(
                    {
                        "model": name,
                        "width": width,
                        "seed": seed,
                        "params": params,
                        "defect_recall": report.defect_recall,
                        "false_alarm_rate": report.false_alarm_rate,
                        "missed_defects": report.missed_defects,
                        "balanced_accuracy": report.balanced_accuracy,
                        "roc_auc": report.roc_auc,
                    }
                )

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "sweep.csv"), index=False)

    summary = (
        df.groupby(["model", "width"])
        .agg(
            params=("params", "first"),
            defect_recall_mean=("defect_recall", "mean"),
            defect_recall_std=("defect_recall", "std"),
            missed_mean=("missed_defects", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    print("\n" + summary.to_string(index=False))
    print(f"\nwrote {os.path.join(args.output_dir, 'sweep.csv')}")


def cmd_leakage_check(args) -> None:
    """Train once on a group-aware split and once on the original ordered split.

    The gap between the two test scores is the size of the illusion the
    original pipeline was reporting.
    """
    from .data import build_index as _bi

    images_dir, csv_path = _resolve_paths(args)
    index = _bi(csv_path)

    results = {}
    for mode in ("grouped", "ordered_leaky"):
        set_seed(args.seed)
        if mode == "grouped":
            splits = group_split(index, seed=args.seed)
        else:
            # Reproduce the original: walk CSV order, fill train, then valid, then test.
            ordered = index.copy()
            n = len(ordered)
            bounds = (int(n * 0.6), int(n * 0.8))
            ordered["split"] = [
                "train" if i < bounds[0] else "valid" if i < bounds[1] else "test"
                for i in range(n)
            ]
            splits = {k: v.reset_index(drop=True) for k, v in ordered.groupby("split")}

        loaders = make_loaders(
            splits, images_dir, batch_size=args.batch_size,
            augment=not args.no_augment, seed=args.seed,
        )
        cfg = TrainConfig(
            model=args.model, width=args.width, epochs=args.epochs,
            lr=args.lr, seed=args.seed, output_dir=args.output_dir,
        )
        model = build_model(args.model, width=args.width)
        model, _ = train_model(model, loaders, cfg, run_name=f"{args.model}_{mode}")

        preds = collect_predictions(model, loaders["test"])
        report = evaluate(preds)
        results[mode] = report.to_dict()
        print()
        print(format_report(report, f"{args.model} [{mode}]"))

    delta = results["ordered_leaky"]["defect_recall"] - results["grouped"]["defect_recall"]
    print(f"\ndefect recall inflated by {delta:+.4f} under the original ordered split")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "leakage_check.json"), "w") as fh:
        json.dump(results, fh, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial biscuit defect detection")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--data-root", default=None, help="dir containing Images/ and Annotations.csv")
        p.add_argument("--model", default="se_resnet", choices=["cnn", "resnet", "se_resnet"])
        p.add_argument("--width", type=int, default=64)
        p.add_argument("--epochs", type=int, default=20)
        p.add_argument("--lr", type=float, default=5e-4)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--no-augment", action="store_true")
        p.add_argument("--originals-only-train", action="store_true")
        p.add_argument("--output-dir", default="runs")

    p_train = sub.add_parser("train"); common(p_train); p_train.set_defaults(func=cmd_train)

    p_sweep = sub.add_parser("sweep"); common(p_sweep)
    p_sweep.add_argument("--widths", type=int, nargs="+", default=[8, 64])
    p_sweep.add_argument("--models", nargs="+", default=["cnn", "resnet", "se_resnet"])
    p_sweep.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p_sweep.add_argument("--quiet", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    p_leak = sub.add_parser("leakage-check"); common(p_leak); p_leak.set_defaults(func=cmd_leakage_check)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
