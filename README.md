# Industrial Biscuit Defect Detection — CNN vs ResNet vs SE-ResNet

Binary defect classification (`ok` / `nok`) on the [Industry Biscuit dataset](https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset), comparing three architectures implemented from scratch in PyTorch: a plain CNN, a residual network with identity and projection shortcuts, and a Squeeze-and-Excitation ResNet.

Coursework project, MSc Artificial Intelligence for Science and Technology, University of Milano-Bicocca / Milan, second semester.

---

## The main finding is a bug in my own evaluation

The first version of this project reported **100% precision, recall and F1 on the test set for all three architectures**. That number is not real, and the reason is worth more than the result.

The dataset ships 4,900 images. They are not 4,900 independent samples — they are **1,225 base images, each with three augmented variants** (1,225 × 4 = 4,900 exactly). My original split walked the annotation CSV in order, filling train, then validation, then test. Because the variants sit at fixed offsets in that CSV, **augmented copies of the same physical biscuit ended up on both sides of the train/test boundary.**

A model does not need to learn what a defect looks like to score well under those conditions. It only needs to recognise an image it has already memorised, slightly rotated.

Three different architectures all reaching a perfect score is the tell. On a real industrial vision task that does not happen; it means the test set is not testing anything.

```
Original split (leaky)                 Group-aware split (this repo)
─────────────────────────              ─────────────────────────────
CSV row order →                        base image + its 3 variants
                                       always travel together
train  [base_412 ............]         train  [base_412, aug_412_0..2]
valid  [aug_412_0, aug_412_1]  ✗       valid  [base_87,  aug_87_0..2 ]
test   [aug_412_2 ..........]  ✗       test   [base_903, aug_903_0..2]
        same biscuit, three splits             no biscuit spans a split
```

`src/data.py` now splits at the **group** level and asserts that no group id appears in two splits. `tests/test_pipeline.py::TestSplitIsLeakageFree` fails if that guarantee ever breaks, and `test_ordered_split_would_have_leaked` reproduces the original bug so the regression stays visible.

**Update (2026-08-11): `leakage-check` has been run.** SE-ResNet, width=8, seed=42, single run, CPU. See [Leakage-check results](#leakage-check-results-verified) below for the measured gap. The full `width=64` sweep across all three architectures on the corrected split has **not** been re-run yet — the table in the next section is still the original leaky-split result and should be read as an upper bound.

---

## Leakage-check results (verified)

`python -m src.main leakage-check --model se_resnet --width 8` trains the identical model once on the group-aware split and once on a reproduction of the original ordered (leaky) split, then reports both on their respective test sets.

| Metric | Grouped (correct) | Ordered (leaky, reproduced) |
|---|---:|---:|
| Defect recall | 0.9833 | 1.0000 |
| Defect precision | 0.9983 | 0.7459 |
| Defect F1 | 0.9908 | 0.8545 |
| False alarm rate | 0.26% | **100%** |
| Balanced accuracy | 0.9904 | 0.5000 |
| ROC AUC | 0.9978 | 0.9975 |
| Missed defects / false alarms | 10 / 1 | 0 / 249 |
| Confusion matrix | TN 379, FP 1, FN 10, TP 590 | TN 0, FP 249, FN 0, TP 731 |

The script's own summary line — `defect recall inflated by +0.0167` — understates what's actually going on. Under the leaky split the model **collapsed to predicting "defect" for every single test image** (249/249 false alarms, balanced accuracy exactly 0.5), while still posting a perfect 1.0000 recall and a 0.9975 ROC AUC. Recall and AUC alone make the leaky-split model look as good as, or better than, the honest one; balanced accuracy and false alarm rate are what expose that it isn't discriminating between classes at all. That's a sharper version of the argument above: leakage doesn't just inflate a metric, it can hide a completely degenerate model behind the metrics people commonly skim.

The grouped split, by contrast, produced a real, working model: 98.3% defect recall with a single false alarm across 980 test images.

Training details: both runs allowed up to 20 epochs with early stopping. The grouped run used the full 20 (best val loss 0.0227 @ epoch 20, 12.7 min on CPU). The leaky run early-stopped at epoch 9 (best val loss 0.0014 @ epoch 4, 5.6 min) — hitting a near-zero validation loss almost immediately is itself a symptom of the same leakage contaminating the validation split, not just the test split.

Environment: Python 3.11.4, PyTorch 2.13.0, CPU only (no GPU available). Full test suite (`python -m pytest tests/ -q`) passes: 25/25.

---

## Original results (leaky split — do not cite these)

Test set, weighted-average metrics, single seed. Preserved for the record.

### Narrow models (`width=8`)

| Model | Params | Best val loss | Test F1 (weighted) |
|---|---:|---:|---:|
| Basic CNN | 808,946 | 0.0512 | 0.9854 |
| Base ResNet | 25,770 | 0.0248 | **0.9968** |
| SE-ResNet | 27,050 | **0.0062** | 0.9903 |

### Wide models (`width=64`)

| Model | Params | Best val loss | Test F1 (weighted) |
|---|---:|---:|---:|
| Basic CNN | 51,751,810 | 0.0213 | 1.0000 |
| Base ResNet | 1,560,898 | **0.0180** | 1.0000 |
| SE-ResNet | 1,572,126 | 0.0216 | 1.0000 |

Two things are visible even through the leakage:

**Validation loss and test F1 disagree.** At `width=8`, SE-ResNet has 4× the validation-loss advantage over the plain ResNet but a *worse* test F1. On a ~980-image test set at 99% accuracy, the gap between 0.9968 and 0.9903 is about six images. With one seed and no confidence interval, that ranking is not distinguishable from noise. `bootstrap_ci` in `src/evaluate.py` now quantifies this instead of leaving it to the eye.

**The plain CNN is a parameter disaster.** At `width=64` it needs 51.8M parameters against the ResNet's 1.6M — 33× more — because it flattens a 256×28×28 feature map straight into a linear layer. That one layer is 97% of the model. The residual networks reach the same accuracy with global average pooling instead. SE adds channel attention for **under 2% parameter overhead** (1,560,898 → 1,572,126), which is the honest argument for it here, independent of the contaminated accuracy numbers.

---

## What changed from the original submission

| Problem | Fix |
|---|---|
| Augmented variants leaked across splits | Group-aware split; assertion + regression test |
| Two near-identical notebooks (`small_…`, `…_BIG_…`) | One `width` parameter; verified to reproduce both parameter counts exactly |
| `average="weighted"` hid defect recall | Defect is the positive class; per-class metrics, confusion matrix, missed-defect counts |
| No seeds anywhere | `set_seed` covers Python, NumPy, torch, CUDA, DataLoader workers |
| No augmentation (relied on the dataset's baked-in variants) | Live train-only augmentation; `--originals-only-train` isolates the contribution |
| Single seed, no error bars | Bootstrap CIs; `sweep` command runs multiple seeds |
| 4,900 JPEGs re-encoded to disk each run | Index the source directory directly |
| Fixed 20–30 epochs past convergence | Early stopping on validation loss |
| Threshold hard-coded at argmax | Probabilities retained; `threshold_sweep` picks an operating point from a defect-recall floor |

On that last point — the metric change matters most for the application. A missed defect ships a bad unit to a customer; a false alarm discards a good one. Those costs are not symmetric, and a weighted average treats them as if they were. `TestMetrics::test_weighted_average_hides_what_per_class_reveals` constructs a case scoring 0.85+ weighted F1 while catching **zero** defects.

---

## Usage

```bash
pip install -r requirements.txt

# Train a single model
python -m src.main train --model se_resnet --width 64 --epochs 20

# Width sweep across seeds — replaces the two original notebooks
python -m src.main sweep --widths 8 64 --seeds 0 1 2

# Measure how much the original split inflated the scores
python -m src.main leakage-check --model se_resnet --width 8
```

Pass `--data-root` to point at an existing copy of the dataset; otherwise it is fetched with `kagglehub`.

```bash
python -m pytest tests/ -q     # 25 tests
```

## Layout

```
src/data.py       group-aware splitting, dataset, transforms
src/models.py     BasicCNN / ResNet / SE-ResNet, unified width parameter
src/train.py      seeded training loop, early stopping, best-checkpoint restore
src/evaluate.py   defect-recall metrics, confusion matrix, bootstrap CIs, threshold sweep
src/main.py       train / sweep / leakage-check
tests/            split integrity, model shapes, metric behaviour
```

## Known limitations

- **The corrected `width=64` sweep across all three architectures is not in yet.** The [leakage-check results](#leakage-check-results-verified) confirm the split fix works and quantify the leakage on one config (SE-ResNet, width=8, single seed); the "Original results" table below it is still the uncorrected leaky-split run.
- The 1,225 base images may themselves come from a smaller number of production runs. If several base images show the same physical biscuit under different lighting, grouping by base image is still too coarse and a session-level split would be needed. I have not verified this against the source data.
- Defect type (`Defect_Shape`, `Defect_Object`, `Defect_Color`) is collapsed to binary. Per-type recall would show which defect classes the models actually struggle with — the binary framing hides that.
- No calibration analysis. The threshold sweep assumes probabilities are meaningful; temperature scaling has not been checked.

## Dataset

Industry Biscuit Cookie Dataset — 4,900 images (1,225 base + 3 augmented variants each), 1,896 `ok` and 3,004 `nok`. Defects are shape, foreign object, and colour. Note the imbalance runs toward defects, which is unusual for production data and suggests deliberate collection rather than a natural line sample.
