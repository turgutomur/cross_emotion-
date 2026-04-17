# Cross-Dataset Emotion Classification

Code for the paper *"Cross-Dataset Emotion Classification under Distribution Shift and Class Imbalance"* — targeting EMNLP 2026 (ARR May cycle).

## Research Plan

**Goal.** Unified Ekman-6 emotion classification across three heterogeneous datasets (GoEmotions, ISEAR, WASSA-21), addressing two coupled problems:
- **Distribution shift** via domain-adversarial training (DANN, CDAN)
- **Class imbalance** via focal loss variants

**Evaluation Protocols.**
- *Protocol A — Mixed*: Train on union of all three datasets, test on held-out mixture.
- *Protocol B — Leave-One-Dataset-Out (LODO)*: Train on two datasets, test on the third. Three transfer scenarios.

**Compared Methods.**
| # | Method | Domain Adaptation | Loss |
|---|--------|:-:|:-:|
| 1 | Source-only | — | CE |
| 2 | Mixed training | — | CE |
| 3 | DANN | ✓ | CE |
| 4 | CDAN | ✓ (conditional) | CE |
| 5 | DANN + Focal (★) | ✓ | Focal |
| 6 | CDAN + Focal (★) | ✓ (conditional) | Focal |

★ = our contribution (joint handling of shift + imbalance).

**Rigor.** All configurations run with ≥3 random seeds; best 2 configurations run with 5 seeds. Statistical significance via paired bootstrap (1000 resamples). Results reported as mean ± std with p-values.

## Compute Budget

- **Primary**: Kaggle free tier (~30 hr/week T4) → ~165 hr over 5.5 weeks
- **Secondary**: Colab Pro (~150 CU total for L4/A100 runs)
- **Backbone**: DeBERTa-v3-base (184M) for main experiments; -large for final validation only

## Timeline (EMNLP 2026 ARR May 25 deadline)

- Week 1: Project scaffold, data loaders, Ekman-6 mapping, LODO splits, sanity check
- Week 2: Source-only + Mixed baselines (3 seeds, Protocol A + B)
- Week 3: DANN + CDAN implementations
- Week 4: Focal loss integration, full LODO matrix
- Week 5: DeBERTa-large validation, statistical tests, visualizations
- Week 6-7: Paper writing

## Directory Layout

```
cross_emotion/
├── configs/              # YAML configs per experiment
├── src/
│   ├── data/            # Dataset loaders, Ekman mapping, LODO splits
│   ├── models/          # Backbone, DANN, CDAN architectures
│   ├── training/        # Trainer, loss functions
│   ├── evaluation/      # Metrics, bootstrap significance tests
│   └── utils/           # Seed control, logging
├── scripts/              # Entry points
├── tests/                # Unit tests for data integrity
├── notebooks/            # Analysis, visualizations
└── outputs/              # Logs, checkpoints, result CSVs
```

## Known Limitations (pre-registered)

- **ISEAR has no `surprise` class.** In LODO with ISEAR as target, surprise F1 is undefined; reported separately.
- **GoEmotions originally multi-label.** We use only single-labeled instances or multi-labeled ones whose labels collapse to a single Ekman-6 class.
- **Shame/guilt (ISEAR) excluded** rather than mapped to sadness (avoids contested mapping).
