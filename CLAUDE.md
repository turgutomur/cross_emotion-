# cross_emotion — Claude Code Briefing

Short, authoritative context for any Claude Code session opened in this
repository. If you need more depth, the two canonical sources of truth are
`README.md` (scope + timeline) and `docs/research_notes.md` (genesis,
rationale, risks). Read those two files before any non-trivial change.

## What this project is

Cross-dataset Ekman-6 emotion classification targeting **EMNLP 2026 ARR May
cycle** (deadline: 2026-05-25, long paper). The scientific claim is a joint
treatment of two coupled problems — distribution shift across three
heterogeneous emotion datasets (GoEmotions, ISEAR, WASSA-21) and class
imbalance — using domain-adversarial training (DANN, CDAN) combined with
focal loss variants. Six methods are compared under two evaluation protocols
(Mixed and Leave-One-Dataset-Out).

The project pivoted from a Sakarya University senior thesis whose original
"Length-Aware Dynamic Routing" model suffered routing collapse and negative
synergy between gating and DANN. Salvaged: working DANN implementation,
three-dataset combination, evaluation framework.

## Pre-registered decisions — DO NOT SILENTLY CHANGE

These commitments are part of the paper's scientific contract with reviewers
and must stay stable. Changing any of them requires an explicit discussion
with the user, not a unilateral "improvement".

**Label space.** Ekman-6 canonical order defined in
`src/data/ekman_mapping.py`: `anger, disgust, fear, joy, sadness, surprise`.
`NUM_LABELS == 6`.

**GoEmotions mapping.** Google's official `ekman_mapping.json` (27 → 6,
neutral dropped). Multi-label examples are kept only when all original
labels collapse to the same Ekman class; otherwise dropped
(`goemotions_strict_single_ekman: true`). Never silently flip this flag.

**ISEAR mapping.** `shame` and `guilt` are EXCLUDED — not mapped to sadness.
Rationale: the shame→sadness mapping is contested in the literature and
would introduce label noise reviewers can flag. Consequence: ISEAR
contributes 5 Ekman classes. `surprise` is absent from the original ISEAR
dataset — known limitation, reported honestly.

**WASSA-21 mapping.** Already Ekman-6, direct use. Any `neutral` rows that
appear in certain distributions are dropped.

**Seeds + statistics.** Minimum 3 seeds (`42, 123, 456`) per configuration;
5 seeds for the final best-two. Paired bootstrap with 1000 resamples, seed
42. Early stopping patience=3 on validation macro-F1. These numbers live in
`configs/default.yaml` — edit the YAML, do not hard-code alternatives.

## Coding conventions

Code and comments in **English**. Conversation with the user in **Turkish**.
Type hints throughout. Docstrings explain *why* (rationale, trade-offs),
not just *what*. When in doubt about style, mirror `src/data/builder.py`
and `src/data/ekman_mapping.py` — those are the reference files.

No bulleted lists in paper drafts (plain prose, full sentences). This
CLAUDE.md uses lists because it is infrastructure, not a paper artifact.

Configs are the single source of truth for hyperparameters. Anything
tunable (learning rate, batch size, dropout, lambda schedules) belongs in
`configs/default.yaml` or a method-specific YAML, not in Python literals.

## Repo layout

```
configs/             YAML per experiment (default.yaml is canonical)
src/data/            Loaders, Ekman mapping, LODO/Mixed protocols
src/models/          Backbone, heads, DANN, CDAN
src/training/        Trainer, losses, optimizer scheduling
src/evaluation/      Metrics, paired bootstrap
src/utils/           Seed control, logging
scripts/             Entry points (sanity_check.py, train.py, eval.py)
tests/               Unit tests — run with `pytest tests/`
notebooks/           Analysis, visualisations
outputs/             Logs, checkpoints, result CSVs (gitignored)
data/raw/            Raw datasets — gitignored, user-managed
```

## Current state (as of 2026-04-23)

**Done (Weeks 1-2, Apr 14-23):**
- Week 1: Data loaders for all three datasets with pre-registered Ekman
  mapping; `EmotionExample` schema, PyTorch dataset+collator; LODO/Mixed
  protocol builders; sanity-check script.
- Week 1 (data quirk): ISEAR loader handles canonical SPSS numeric codes
  1-7 via `canonicalize_isear_label` in `ekman_mapping.py` — pre-registered
  exclusion of shame (6) and guilt (7) preserved across formats. WASSA
  loader accepts `track-1-essay-empathy-train.tsv` filename variants and
  falls back to stratified split when dev/test missing or unlabeled.
- Week 2: `src/models/backbone.py` (DeBERTa-v3-base wrapper, fp32-stable
  for AMP — encoder must NOT be cast to fp16 manually; AMP autocast handles
  precision, GradScaler unscales fp32 grads). `src/models/classifier.py`
  (CE head + composite). `src/training/trainer.py` (fp16 AMP, grad-accum 2,
  early stopping patience=3 on val macro-F1, per-seed CSV append — see
  `write_results_csv` for the append-with-header-reuse pattern).

**Week 2 baseline numbers (CE only, no domain adaptation):**
- Mixed/mixed, 3 seeds: test_macro_f1 = **0.7102 ± 0.005**
- LODO target=goemotions, seed 42: test = **0.2977**  (gap from val: 47.5)
- LODO target=isear,      seed 42: test = **0.5145**  (gap: 19.8)
- LODO target=wassa21,    seed 42: test = **0.4466**  (gap: 27.9)
- LODO mean test ≈ **0.42** — about 29 points below Mixed. This is the
  bar DANN/CDAN must close.

**In progress (Week 3, Apr 28 - May 4, started early):**
- `src/models/dann.py` — Gradient Reversal Layer + 3-class domain
  discriminator with sigmoid lambda annealing.
- `src/models/cdan.py` — Conditional DANN (class-conditional alignment).
- `src/training/trainer.py` joint task+domain loss path.
- Run DANN/CDAN on Mixed (3 seeds) + each LODO target (1 seed first,
  expand to 3 if promising).

**Planned (Weeks 4-5):**
- Week 4: Focal loss variants (DANN+Focal, CDAN+Focal), full LODO matrix
  expanded to 3 seeds per cell.
- Week 5: DeBERTa-large validation runs (A100), paired bootstrap
  significance tests, visualisations.

**Paper writing:** Weeks 6-7 (May 19-25).

**Compute path note:** Primary execution is Colab Pro on L4 (~5 min/epoch
for Mixed, ~2 min/epoch for LODO). Outputs persist to
`/content/drive/MyDrive/cross_emotion_data/outputs/`. Always pass
`--output-dir /content/drive/MyDrive/cross_emotion_data/outputs` so CSVs
land on Drive, not the ephemeral Colab disk.

## Compute budget

Primary: Kaggle free tier, ~30 h/week T4. Secondary: Colab Pro, ~100 CU/month
(mostly L4, A100 reserved for the DeBERTa-large validation pass). Main
experiments use DeBERTa-v3-base (184M); -large is for final validation only.

Training defaults: batch_size=16, grad_accum=2 (effective 32), fp16=true,
encoder_lr=1e-5, head_lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, 15 epochs.

## How to verify changes

Before finishing any task that touches data or models, run:

```bash
# Data pipeline end-to-end
python scripts/sanity_check.py
# Or if only GoEmotions is available locally:
python scripts/sanity_check.py --goemotions-only

# Unit tests
pytest tests/ -v
```

For model / training changes, also do a single-epoch smoke run on a 500-row
subset to confirm loss decreases before launching multi-seed jobs.

## Paths not to touch

`data/raw/` — raw datasets, gitignored, user-managed. Do not commit these
files; they are large, some are licensed.

`outputs/` — experiment artefacts, generated by training runs. Do not
manually edit checkpoints or result CSVs committed here.

## When in doubt

Ask before making destructive changes (deleting files, rewriting loaders,
changing label mappings, touching seeds). The user prefers a one-line
clarifying question over an unwanted refactor.
