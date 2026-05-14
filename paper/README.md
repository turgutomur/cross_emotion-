# Paper Directory — EMNLP 2026 ARR May Submission

LaTeX source for the cross-dataset emotion classification paper.

## Structure

```
paper/
├── main.tex                  # main document; includes everything below
├── references.bib            # bibliography
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methods.tex
│   ├── experimental_setup.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── conclusion.tex
│   ├── limitations.tex
│   ├── ethics.tex
│   └── appendix.tex
└── README.md                 # this file
```

Figures are read from `../outputs/figures/*.pdf` (relative path),
which are produced by `scripts/make_figures.py` on Colab and stored
on Drive. For local builds, ensure the figures are copied into
`outputs/figures/` first or adjust `\includegraphics` paths.

## ACL/ARR style files

The template uses `acl_latex.sty` and `acl_natbib.bst` from the
official ACL style files repository:
[https://github.com/acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)

Download the latest May 2026 ARR template release and drop these two
files into this directory:

```
paper/acl_latex.sty
paper/acl_natbib.bst
```

If you don't have them yet, `main.tex` will fail to compile with the
ACL package — comment out the `\usepackage{acl_latex}` line for a
plain-article fallback while drafting.

## Build

Recommended (handles bibtex automatically):

```bash
cd paper/
latexmk -pdf main.tex
```

Manual:

```bash
cd paper/
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## TODO macros

Every `\todo{...}` and `\note{...}` macro renders in red/blue in the
output PDF. Grep before submission:

```bash
grep -rn "TODO\|todo{" sections/
```

All TODOs must be resolved or moved to the appendix.

## Page limit

ARR EMNLP long paper: **8 pages main body + unlimited appendix +
references**. The Limitations and Ethics sections are required and
count outside the 8-page limit.

## Numbers source-of-truth

All numerical claims in the paper trace to:

- `docs/results_final.md` — consolidated tables and findings
- `outputs/results/*.csv` and `outputs/**/results/*.csv` — raw per-seed
  test F1 numbers
- `outputs/bootstrap/pairwise_pvalues.csv` — paired bootstrap p-values

When updating numbers, update `docs/results_final.md` first, then
propagate to the LaTeX via the `\newcommand{\focal*}` macros in
`main.tex` so each value appears in exactly one place.

## Anonymity

Author block currently set to `Anonymous Submission` for ARR review.
De-anonymize on acceptance.
