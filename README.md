# Pairing Center Prediction

Predicting meiotic **pairing-center (PC) DNA motifs** in *Caenorhabditis briggsae* (and
*C. remanei*), using the well-annotated *C. elegans* system as a reference.

## Background

During meiosis, homologous chromosomes must find their partner chromosome in a process called homologous pairing. In *C. elegans* this is
organized by **pairing centers**, regions near one end of each chromosome, bound by four
zinc-finger protein paralogs (**ZIM-1, ZIM-2, ZIM-3, HIM-8**) that each recognize a short
DNA motif. All four motifs share a conserved **`TTGG`…`TG` framework** and differ mainly in the
spacer between those sub-sites. The motifs in other *Caenorhabditis* species are unknown; this
project predicts them.

## The scientific arc

Two fundamentally different strategies were tried, and the contrast is the main finding:

1. **Predict the motif from the protein.** Frozen ESM-2 embeddings of each
   ZIM/HIM-8 protein → a small learned head → the DNA motif. **This does not work here**, and we
   show *why* with a baseline hierarchy (conserved floor → nearest-neighbour → learned head):
   none beat the floor. The reasons are structural — only **4 labelled proteins**, and (per Li
   et al. 2024) the specificity determinant is a **CTD conformation** that a sequence embedding
   cannot read (the same reason AlphaFold-3 fails on this family). A rigorous negative result.

2. **Discover the motif directly from the genome.** A pairing center is a **dense tandem array**
   of its motif near a chromosome end, so the answer is already written thousands of times in the DNA.
   Locating those arrays and reading out a motif (PWM + EM refinement) **works**, because the
   data is abundant and directly observable rather than learned from four examples. The problem switches from being a seq2seq problem to      a data labeling one.

### Headline predictions (C. briggsae)

De novo discovery, reproduced across two independent assemblies (AF16 + chromosome-level
QX1410), yields chromosome-specific, tandem, framework-following candidate motifs. The
gold-standard calls (well-defined, reproduced, and agreeing with the expected paralog):

| Chromosome | Predicted binder | Motif | Note |
|---|---|---|---|
| III | Cbr-ZIM-1 | `TTGGTCTGCTAATTAT` | near-exact match to the *C. elegans* ZIM-1 motif |
| X | Cbr-HIM-8 | `TTGGTAGTGGTTCCGC` | novel candidate for the *C. briggsae* X pairing center |
| IV | Cbr-ZIM-3 | `TTGGGTCATGACCTAG` | tandem, chromosome-specific |

The same TTGG-framework ZIM-1 motif is recovered at chromosomes II/III across **three species**
(*C. elegans*, *C. briggsae*, *C. remanei*) — evidence the motif is evolutionarily conserved.
These are computational predictions to be confirmed experimentally (e.g. ChIP-seq).

### Guarding against circular reasoning

Because searching for `TTGG` could be self-fulfilling, the repo includes drift-safe controls:
an **anchor-free** discovery run (no TTGG assumption) and a **recognition-residue conservation**
analysis that decides, per protein, whether the TTGG anchor is justified (conserved binder) or
whether the motif may have drifted (diverged binder, e.g. HIM-8). See `RESULTS.md` for the full
reasoning.

## Repository layout

```
src/                 reusable modules (no side effects on import)
  data_io, genome_qc, orthologs, motifs, embeddings, metrics, zinc_fingers, paths
eda.py               genome characterization, motif scans, terminal k-mer enrichment
build_selex_pwms.py  SELEX oligos -> PWMs
find_orthologs*.py   C. briggsae ortholog identification (Python + BLAST)
build_gene_tree.py   ZIM/HIM-8 family gene tree
compute_embeddings.py / compare_domain_embeddings.py / recognition_baseline.py   ESM-2 features
run_baselines.py     Baselines 1-2 (conserved floor, nearest-neighbour)
pretrain_jaspar.py / train_head.py   Baseline 3 (JASPAR-pretrained learned head)
tier23_evaluation.py Tier 2/3 biological-plausibility evaluation
discover_motifs.py   de novo motif discovery (TTGG framework), AF16 + QX1410 + reconcile
discover_denovo.py   drift-safe control: anchor-free, specificity-selected discovery
discover_anchored.py drift-aware sheet (TTGG vs anchor-free + conservation)
crem_ttgg_check.py / compare_runs.py   C. remanei cross-species replication
data/
  raw/               genomes / annotations / SELEX (NOT in git; download from WormBase)
  processed/         all generated results, organized by method (see data/processed/README.md)
RESULTS.md           layer-by-layer results and interpretation
REFERENCES.md        reading list (methods + biology)
```

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Then run the scripts in the order documented in **`RESULTS.md`** (from `eda.py` through the
discovery scripts). All outputs are written into categorized subfolders under `data/processed/`
via `src/paths.py`. The full-pipeline reproduce commands and expected numbers are in `RESULTS.md`.

## Data sources

Genomes/annotations are large and **not committed** — download from WormBase ParaSite (WBPS19):
*C. elegans* `PRJNA13758`, *C. briggsae* `PRJNA10731` (AF16) and `QX1410_PRJNA784955`
(chromosome-level), *C. remanei* `PRJNA577507`. SELEX training data: Phillips et al. 2009,
Table S3. Structural reference: Li et al. 2024, *Nat Commun*.

## Documentation

- **`RESULTS.md`** — detailed, layer-by-layer results, figures, and interpretation.
- **`REFERENCES.md`** — methods and biology reading list.
- **`data/processed/README.md`** — index of every generated artifact by method.
