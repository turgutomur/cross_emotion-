# Research Notes — Cross-Dataset Emotion Classification

## Project Genesis

This project pivots from a senior thesis (Sakarya University, SWE402) that proposed
a "Length-Aware Dynamic Routing" model for emotion classification. The original model
used DeBERTa-v3-base + CNN/BiLSTM branches + gating mechanism + DANN.

### Why the Pivot

The original approach had fundamental issues that made it unpublishable as-is:

1. **Routing collapse**: The gating mechanism failed to learn length-based routing.
   Gate values for short vs. long texts had separation of only 0.0102;
   domain classification accuracy using gate values was 48.2% (worse than random).

2. **Negative synergy**: Combining gating + DANN degraded performance (–4.90% F1
   vs. expected additive gain). The full model (0.6598) performed worse than
   gating-only (0.6847) and DANN-only (0.6806).

3. **Best config was NOT the proposed model**: Gating-only (without DANN) was best,
   but this contradicts the paper's thesis about joint routing+adaptation.

### What Was Salvageable

- Working DANN implementation (+2.41% improvement in isolation)
- Three-dataset combination (GoEmotions + ISEAR + WASSA-21)
- Evaluation framework (ablation tables, confusion matrices, per-class F1)
- Training pipeline code

## New Direction: Cross-Dataset Emotion Classification

Instead of forcing length-aware routing, we reframe as a **cross-dataset domain
adaptation** problem, which is the natural use case for DANN.

### Core Idea

Three datasets with different text styles but shared Ekman-6 labels represent
three genuine domains. DANN was designed for exactly this type of distribution
shift — not for the artificial "short vs long" domain the thesis used.

### Contribution Claims

1. **Unified benchmark**: GoEmotions + ISEAR + WASSA-21 mapped to Ekman-6 with
   documented preprocessing (important: mapping decisions pre-registered).
2. **Joint handling of shift + imbalance**: Domain adaptation (DANN/CDAN) combined
   with focal loss addresses two coupled problems simultaneously.
3. **Systematic LODO evaluation**: Leave-one-dataset-out protocol provides a
   genuine test of cross-domain generalization.

## Key Design Decisions (Pre-registered)

### Label Mapping

- **GoEmotions**: Google's official ekman_mapping.json (27→6, neutral dropped).
  Multi-label examples: keep if all labels collapse to same Ekman class; drop otherwise.
- **ISEAR**: shame/guilt EXCLUDED (not mapped to sadness). Only 5 Ekman classes present (no surprise).
- **WASSA-21**: Already Ekman-6, direct use.

### Evaluation Protocols

- **Protocol A (Mixed)**: Union of all three train sets → test on union of test sets + per-dataset test.
- **Protocol B (LODO)**: Train on 2 datasets, test on 3rd. Three scenarios.

### Statistical Rigor

- Minimum 3 seeds (42, 123, 456); 5 seeds for final best-2 methods.
- Paired bootstrap test (1000 resamples) for significance claims.
- Report mean ± std for all metrics.
- Early stopping (patience=3) on validation macro-F1.

## Compute Constraints

- **Primary**: Kaggle free tier (~30h T4/week)
- **Secondary**: Colab Pro (~100 CU/month, T4/L4/A100 available)
- **Implication**: DeBERTa-v3-base for main experiments; -large only for final validation of best 2-3 configs on A100.

## Target Venue

- **EMNLP 2026** ARR May cycle (deadline: May 25, 2026)
- Long paper (8 pages)
- Fallback: commit to AACL 2026 or NAACL 2027 with same reviews

## Timeline

- Week 1 (Apr 14-20): Project scaffold, data loaders ← CURRENT
- Week 2 (Apr 21-27): Source-only + Mixed baselines (3 seeds)
- Week 3 (Apr 28-May 4): DANN + CDAN implementations
- Week 4 (May 5-11): Focal loss, full LODO matrix
- Week 5 (May 12-18): DeBERTa-large validation, stats, visualization
- Week 6 (May 19-25): Paper writing, submission

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Mixed training beats DANN | Report honestly; LODO may differentiate |
| Focal loss helps Source-only equally | Include Source-only+Focal as a row |
| ISEAR surprise absence complicates LODO | Compute metrics over present classes only |
| 100 CU/month insufficient | Kaggle as primary; Colab for L4/A100 only |
| Negative results again | Reframe as empirical study with benchmark contribution |
