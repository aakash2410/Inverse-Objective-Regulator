# Paper build

LaTeX source for the IOR system paper, arXiv-ready, multi-file.

## Structure

```
paper/
  main.tex                 master file: preamble, title, abstract, \input includes
  references.bib           bibliography (natbib / plainnat)
  sections/
    00-abstract.tex
    01-introduction.tex
    02-related-work.tex
    03-problem-formulation.tex   full prose
    04-architecture.tex
    05-inference.tex             full prose
    06-divergence-targeting.tex
    07-gym.tex
    08-evaluation.tex
    09-limitations.tex
    10-conclusion.tex
  figures/                 figure artwork (placeholders in text for now)
```

## Build

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or, if available:

```
latexmk -pdf main.tex
```

No local TeX toolchain is required to edit; the source also compiles on Overleaf
and on the arXiv build system.

## Conventions

- British English, active voice, no em-dashes (LaTeX `---` is avoided; it renders
  as an em-dash). Use commas, colons, or parentheses.
- Sections 3 and 5 are full prose. The remainder carry ported outline content;
  unfinished prose and pending numbers are flagged with `\todo{...}` (renders in
  red) so they are visible in the compiled PDF.
- Figures are `\figplaceholder{...}` boxes until artwork is produced.
- Citation keys live in `references.bib`. Every arXiv entry carries a `note` to
  re-verify ID, title, and authors before submission.

## Status

- Drafted prose: abstract, sections 1, 2, 3, 4, 5, 6, 7, 9, 10.
- Pending live-API numbers: section 8 (evaluation). The structural-floor result
  (0.335 mean behaviour-prediction lift) is measured and included; judge and
  ensemble rows await an API run.

## Relationship to the markdown drafts

`outline.md` remains the planning document. `section3.md` and `section5.md` are the
original markdown drafts now superseded by `sections/03-*.tex` and
`sections/05-*.tex`; keep them for diffing or delete them once the LaTeX is the
single source of truth.
