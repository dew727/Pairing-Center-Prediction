# Pairing Center Prediction

Predicting meiotic pairing-center (PC) DNA motifs in *Caenorhabditis briggsae* (and
*C. remanei*), using the well-characterized *C. elegans* system as a reference.

## Background

During meiosis, homologous chromosomes have to find their partner. In *C. elegans* this is
organized by pairing centers, regions near one end of each chromosome that are bound by four
paralogous zinc-finger proteins (ZIM-1, ZIM-2, ZIM-3, HIM-8). Each protein recognizes a short
DNA motif. All four motifs share a conserved `TTGG`...`TG` framework and differ mainly in the
spacer between those two sub-sites. The motifs in other *Caenorhabditis* species are not known,
and this project tries to predict them.

## Approach and main result

Two different strategies were tried, and the contrast between them is the main result.

1. Predict the motif from the protein (machine learning). Frozen ESM-2 embeddings of each
   ZIM/HIM-8 protein feed a small learned head that outputs the DNA motif. This does not work
   here, and the repo shows why with a baseline hierarchy (conserved floor, nearest-neighbour,
   learned head): none of them beat the floor. There are two structural reasons. First, there
   are only 4 labelled proteins. Second, following Li et al. 2024, the specificity determinant
   is a CTD conformation that a sequence embedding cannot read, which is also why AlphaFold-3
   fails on this family. It is reported as a negative result.

2. Discover the motif directly from the genome. A pairing center is a dense tandem array of its
   motif near a chromosome end, so the motif appears in many copies in the genomic sequence.
   Locating those arrays and reading out a motif (a PWM refined by EM) does work, because the
   data is abundant and directly observable rather than learned from four examples.

3. Scale the protein model across species, using (2) to generate the labels (1) lacked. Every
   additional species with a finished annotation contributes its own ZIM/HIM-8 proteins paired
   with the motifs read off its own chromosome ends, so the training set grows with the number
   of genomes instead of being pinned at four. These are *silver* labels — discovered, not
   measured — so they are weighted by confidence, and the four measured *C. elegans* motifs are
   held back as the test set. Implemented and verified on synthetic data; not yet run on the
   real genomes. See `RESULTS.md` Layer 7.

   The bar this has to clear is deliberately not the uniform floor. Because a motif is joined to
   a protein by synteny, a model can score well by learning only which paralog group a protein
   belongs to. So every result is reported against a **group-consensus baseline** that does
   exactly that and ignores the protein embedding entirely; only beating *it* is evidence that
   the protein language model contributed anything.

### Predictions (C. briggsae)

De novo discovery was run on two independent assemblies (AF16 and the chromosome-level QX1410)
and gives chromosome-specific, tandem, framework-following candidate motifs. The strongest
calls, which are well-defined, reproduced on both assemblies, and agree with the expected
paralog:

| Chromosome | Predicted binder | Motif | Note |
|---|---|---|---|
| III | Cbr-ZIM-1 | `TTGGTCTGCTAATTAT` | near-exact match to the *C. elegans* ZIM-1 motif |
| X | Cbr-HIM-8 | `TTGGTAGTGGTTCCGC` | novel candidate for the *C. briggsae* X pairing center |
| IV | Cbr-ZIM-3 | `TTGGGTCATGACCTAG` | tandem, chromosome-specific |

The same TTGG-framework ZIM-1 motif turns up at chromosomes II/III in all three species
(*C. elegans*, *C. briggsae*, *C. remanei*), which is evidence that the motif is
evolutionarily conserved. These are computational predictions and still need experimental
confirmation, for example by ChIP-seq.

### Guarding against circular reasoning

Searching for `TTGG` could be self-fulfilling, so the repo includes two controls: an
anchor-free discovery run that assumes no TTGG at all, and a recognition-residue conservation
analysis that decides, per protein, whether the TTGG anchor is justified (a conserved binder)
or whether the motif may have drifted (a diverged binder, such as HIM-8). See `RESULTS.md`
for the full reasoning.

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
prepare_species.py   annotation-completeness gate: which species are fit to train on
build_multispecies_labels.py   orthologs + per-species discovery -> silver training labels
embed_multispecies.py / train_multispecies.py   the multi-species model (leave-one-species-out)
tests/               synthetic-data tests of the multi-species pipeline
data/
  species_manifest.csv   the species registry + which annotations are complete
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

Then run the scripts in the order given in `RESULTS.md`, from `eda.py` through the discovery
scripts. Outputs are written into categorized subfolders under `data/processed/` via
`src/paths.py`. The full reproduce commands and expected numbers are in `RESULTS.md`.

## Data sources

Genomes and annotations are large and are not committed. Download them from WormBase ParaSite
(WBPS19): *C. elegans* `PRJNA13758`, *C. briggsae* `PRJNA10731` (AF16) and
`QX1410_PRJNA784955` (chromosome-level), *C. remanei* `PRJNA577507`. SELEX training data is
from Phillips et al. 2009, Table S3. The structural reference is Li et al. 2024, *Nat Commun*.

## Documentation

- `RESULTS.md`: detailed, layer-by-layer results, figures, and interpretation.
- `REFERENCES.md`: methods and biology reading list.
- `data/processed/README.md`: index of every generated artifact by method.
