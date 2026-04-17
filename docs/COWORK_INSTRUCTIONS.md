# Cross-Dataset Emotion Classification — EMNLP 2026 Project

## Research plan (short)
- Unified Ekman-6 emotion classification under cross-dataset distribution shift.
- Datasets: GoEmotions, ISEAR, WASSA-21 (all mapped to Ekman-6).
- Methods: Source-only, Mixed, DANN, CDAN, DANN+Focal, CDAN+Focal.
- Protocols: (A) Mixed training/test; (B) Leave-One-Dataset-Out.
- Target venue: EMNLP 2026 ARR May cycle (deadline May 25, 2026). Long paper.

## Label mapping decisions (pre-registered, do not change)
- GoEmotions → Ekman-6: Google's official mapping (27 → 6, neutral dropped).
- ISEAR: shame and guilt EXCLUDED (not mapped to sadness).
- ISEAR has no `surprise` class — known limitation.
- WASSA-21: already Ekman-6, direct use.

## Compute budget
- Primary: Kaggle free tier (30h T4/week).
- Secondary: Colab Pro (~100 CU/month, mostly L4 with A100 reserved for final -large runs).
- Backbone: DeBERTa-v3-base for main runs; -large only for final validation.

## Rigor requirements
- Minimum 3 seeds per configuration; 5 seeds for final table best-two.
- Paired bootstrap (1000 resamples) for significance.
- Per-dataset AND aggregate metrics always reported.
- Early stopping (patience=3) on val macro-F1.

## Current status
- Week 1 scaffold COMPLETE: data loaders, Ekman mapping, protocols, evaluation, tests.
- NEXT: Hafta 2 — Source-only and Mixed baselines.

## What to build next (priority order)
1. src/models/backbone.py — DeBERTa encoder wrapper
2. src/models/classifier.py — Classification head (CE baseline)
3. src/training/trainer.py — Training loop with early stopping
4. src/training/losses.py — Focal loss
5. src/models/dann.py — Gradient Reversal Layer + domain discriminator
6. src/models/cdan.py — Conditional DANN

## Style
- Docstrings on every public function with rationale, not just "what".
- Type hints throughout.
- No bulleted lists in paper drafts; full prose.
- Turkish conversation, English code and comments.

## Key files to read first
- docs/research_notes.md — Full project context and pivot rationale
- README.md — Overview and timeline
- configs/default.yaml — All hyperparameters
- src/data/ekman_mapping.py — Label harmonization (single source of truth)
