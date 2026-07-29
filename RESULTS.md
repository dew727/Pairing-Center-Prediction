# Results & artifacts

Generated artifacts for the three completed layers (data/EDA, SELEX→PWM, ortholog
identification). All files live in `data/processed/` and are reproducible by
re-running the scripts noted below. Genomes/annotations are *not* in git (large);
download from WormBase ParaSite WBPS19: *C. elegans* PRJNA13758, *C. briggsae*
PRJNA10731 (AF16) and QX1410_PRJNA784955 (chromosome-level), *C. remanei* PRJNA577507.

| Script | Produces |
|--------|----------|
| `eda.py` | genome stats, motif scans + density plots, spacing, k-mer enrichment |
| `build_selex_pwms.py` | SELEX → PWMs (`pwms.npz`, `pwms_meta.json`) |
| `find_orthologs.py` | C. briggsae orthologs, pure-Python search (`ortholog_table.csv`) |
| `find_orthologs_blast.py` | same via BLAST+ (`ortholog_table_blast.csv`) — cross-check |
| `build_gene_tree.py` | ZIM/HIM-8 family gene tree (`zim_family_tree.*`) |
| `run_baselines.py` | Baselines 1–2 scored by the locked metric (`baseline_scores.csv`) + C. briggsae predictions (`cb_predicted_motifs_nn.csv`) |
| `pretrain_jaspar.py` | JASPAR C2H2 → PLS projection for pretraining (`jaspar_projection.npz`) |
| `train_head.py` | Baseline 3 learned head, LOPO (`head_scores.csv`, `head_l2_sweep.csv`) |
| `recognition_baseline.py` | biology-informed ZF-recognition-helix features + region-localization test |
| `tier23_evaluation.py` | Tier 2/3 biology of the C. briggsae predictions (`cb_pc_candidates.csv`, `.png`) |
| `discover_motifs.py` | de novo PC-motif discovery from the genome (`discovered_motifs.csv/npz/png`) |

---

## Layer 1 — EDA / genome characterization (`eda.py`)

**Headline figure — `ce_motif_density.png`.** Scanning the C. elegans genome for
the four known motifs reproduces **Figure 2 of Phillips et al. 2009**: each
paralog's motif clusters at a chromosome end, chromosome-specifically — HIM-8 at
the X end, ZIM-1 at II/III, ZIM-3 at I/IV, ZIM-2 at the V end. This is the
critical sanity check that the scanning pipeline is correct.

**`cb_motif_density_elegans_motifs.png`.** The same scan in C. briggsae: the
signal largely collapses, directly supporting the project premise that
C. briggsae uses *different* pairing-center motifs.

Motif-hit counts (whole genome):

| Motif | C. elegans hits | C. briggsae hits (C. elegans motif) |
|-------|----------------:|-----------------------------------:|
| HIM-8 | 290 | 7 |
| ZIM-1 | 395 | 295 |
| ZIM-2 | 1017 | 5 |
| ZIM-3 | 1002 | 8 |

Other artifacts:
- `chromosome_lengths.png`, `gene_counts.png` — assembly overview (C. elegans
  7 seqs / 100.3 Mb; C. briggsae 367 seqs / 108.4 Mb).
- `motif_spacings.png` — spacing between consecutive motif hits vs expected
  tandem-repeat periods.
- `ce_motif_hits.csv`, `cb_motif_hits.csv` — every motif hit (chrom, position, strand).
- `cb_kmer_enrichment.csv` — **1,255 TTGG-containing 12-mers enriched at
  C. briggsae chromosome ends**: the unsupervised candidate pairing-center motifs.

By name, the EDA finds all four C. elegans paralogs (9 proteins incl. isoforms)
but only **Cbr-him-8** in C. briggsae — the zim orthologs are not annotated by
name, which is exactly what motivates Layer 3. HIM-8 C. elegans-vs-C. briggsae
protein identity is **49.1%**.

## Layer 2 — SELEX → PWMs (`build_selex_pwms.py`)

`pwms.npz` (+ `pwms_meta.json` provenance). Four PWMs built from the 2009 SELEX
bound oligos, trusting the source file's whitespace alignment, trimming columns
below 50% coverage, pseudocount 0.25. Each PWM is `(4, L)`, rows A/C/G/T,
columns summing to 1; raw counts and coverage are also stored.

| Motif | Domain | Oligos | Width | Consensus |
|-------|--------|-------:|------:|-----------|
| HIM-8 | ZnF core | 90 | 21 | `GAATTGGCACGGTGCCAAGTC` |
| ZIM-2 | ZnF core | 100 | 21 | `ATCTTGGCAAGGGGCCAAATA` |
| ZIM-3 | ZnF core | 141 | 22 | `GACTTGGCATCGTGCCAAGTCT` |
| HIM-8-CTD | C-terminus | 115 | 23 | `CCCAAGCACTGACCAACCCGCCC` |

The three ZnF-core consensuses show the expected `TTGGCA…TGCCAA` inverted-repeat
binding site; HIM-8-CTD is a distinct site with no TTGG. **No ZIM-1 SELEX data
existed in 2009**, so it has no PWM (its motif will come from the 2024 structure).

## Layer 3 — Ortholog identification + gene tree

`ortholog_table.csv` (Python k-mer + BLOSUM62 search) and
`ortholog_table_blast.csv` (BLAST+ 2.17.0 cross-check). No blastp was installed
and the C. briggsae zims are unnamed, so orthologs were found by homology;
him-8 is anchored to the curated Cbr-him-8, and zim-1/2/3 resolved by a
one-protein-per-gene 1:1 assignment.

| Paralog | C. briggsae ortholog | Confidence |
|---------|----------------------|------------|
| zim-1 | CBG12895 | **High** — both methods agree; BLAST E=3e-108, reciprocal-best-hit |
| him-8 | CBG12898 (Cbr-him-8) | **High** — curated name + reverse-BLAST RBH back to him-8 |
| zim-2 / zim-3 | {CBG12897, CBG29927} | **Low** — methods disagree on which is which |

All four orthologs lie within a **23 kb cluster on C. briggsae chr I**
(10.854–10.877 Mb), mirroring the C. elegans tandem cluster on chr IV — i.e. the
cluster is syntenically conserved.

**Gene tree — `zim_family_tree.png`** (+ `zim_family.nwk`, `zim_family_tree.txt`).
MAFFT alignment of the family (both species) → neighbour-joining (BLOSUM62) →
100× bootstrap. him-8 pairs cross-species with Cbr-him-8 (clean 1:1 ortholog),
while the zim paralogs group **by species** — indicating lineage-specific
expansion, i.e. the zim subfamily is a set of **co-orthologs** with no distinct
zim-2-vs-zim-3 ortholog. (Bootstrap support is low, ~36–42%; the rigorous
follow-up is a maximum-likelihood tree of the zinc-finger domain only.)

## Layer 4 — Baselines & the learned head (internal LOPO eval)

Every method is scored by the one locked metric (`src/metrics.py`): mean per-column
distance on the **variable** motif positions only (the conserved TTGG/TG anchors are
excluded), rescaled to a **skill** score where 0 = no better than a blind uniform
guess and 1 = the true motif exactly. Evaluation is leave-one-paralog-out (LOPO)
over the four C. elegans paralogs. Cross-paralog predictions are aligned to the
held-out protein's frame by the shared TTGG **and** TG anchors (`anchored_transfer`
in `src/motifs.py`), TG-registered.

**Headline result: no baseline beats the conserved-consensus floor on internal LOPO.**

| Baseline | Mean skill | Reading |
|----------|-----------:|---------|
| 1 — conserved-consensus (floor) | **0.000** | predicts uniform at every variable position |
| 2 — nearest-neighbour (domain emb., mean-centered) | **−0.31** | *below* the floor |
| 3 — learned head, fine-tune only (L2=10) | **−0.12** | below floor; → 0.000 only as L2 → ∞ |
| 3 — learned head, JASPAR-pretrained (L2=10) | **−0.13** | pretraining does **not** help |

Why this is expected, not a failure of implementation:

- **The graded positions are, by construction, the ones that differ between
  paralogs.** So the nearest-neighbour rule (copy another paralog's motif) places
  *confident wrong* letters at exactly the graded positions — worse than the floor's
  uniform guess (a confident wrong column is ~1.37 away vs. 0.833 for uniform).
  `baseline_scores.csv`. Centering behaves as documented — it collapses the ESM-2
  shared component (cosines 0.99 → 0.02–0.21) and changes which neighbour HIM-8/ZIM-3
  pick — but no retrieval choice can win when the target letters are unique to the
  held-out paralog.
- **The learned head cannot generalise from three examples to a held-out paralog's
  unique letters.** The L2 sweep (`head_l2_sweep.csv`) is monotonic: weak
  regularisation memorises noise (skill ≈ −0.17), strong regularisation falls back
  to predicting the mean training composition and approaches the floor **from
  below** (skill → 0⁻), but never exceeds it at any setting.
- **JASPAR pretraining is real but does not rescue the internal eval.** The PLS
  projection (1280 → 12) learned from **434** C2H2 zinc-finger factors captures
  genuine motif signal (in-sample motif-feature R² = 0.37), yet the pretrained head
  tracks the fine-tune-only head to within ±0.01 skill across the whole L2 sweep.

Interpretation: four paralogs with hyper-specific variable positions are **below the
threshold at which internal LOPO can demonstrate protein→motif generalisation**. This
does not doom the project — it relocates the evidence. The case for any predicted
motif must come from the **Tier 2/3 biological evaluation** (does the predicted
C. briggsae motif cluster at a chromosome end, chromosome-specifically, with
tandem-repeat spacing, and agree with the model-free terminal k-mer enrichment?) and,
ultimately, from far more training data (SELEX/ChIP across more species). A caveat on
the negative result itself: with n=4, all skill numbers carry large variance; they
bound what LOPO can *show*, not necessarily what a scaled-up model could achieve.

**C. briggsae predictions (`cb_predicted_motifs_nn.csv`).** The nearest-neighbour rule
applied cross-species (C. elegans average used for centering; the C. briggsae proteins
are disjoint from it, so no leakage) predicts each ortholog binds its most-similar
C. elegans paralog's motif. 3 of 4 orthologs select their own paralog as nearest
(Cbr-ZIM-1→ZIM-1 cos 0.51, Cbr-ZIM-3→ZIM-3 cos 0.51, Cbr-ZIM-2→ZIM-2 cos 0.30);
only Cbr-HIM-8 pulls toward ZIM-1 (cos 0.31) rather than HIM-8 — consistent with
HIM-8 being the divergent short paralog seen throughout the ortholog analysis. These
are the deliverable predictions to carry into the Tier 2/3 checks; they have no
internal skill score because no C. briggsae ground truth exists.

**Leakage discipline.** LOPO centering means are fit on the training fold only; the
JASPAR pretraining set excludes any nematode / zim / him-8 factor by name and species.

---

## Layer 5 — Biology-informed features & the Tier 2/3 evaluation

**Recognition-helix features (`recognition_baseline.py`, `src/zinc_fingers.py`).**
Following Liu 2009 (gene conversion homogenises the N-terminus; specificity sits in the
zinc fingers) and Li et al. 2024 (crystal structures), we embed *only* the two atypical
C2H2 fingers' recognition helices instead of mean-pooling the whole domain. This sharpens
nearest-neighbour discrimination substantially — **mean skill −0.31 → −0.077** — and
recovers the correct Cbr-HIM-8 → HIM-8 assignment that whole-domain mean-pooling missed,
but it still does not beat the conserved floor. A feature-localization test (nearest-
neighbour skill: recognition-helices −0.077, CTD-only −0.322, whole ZF1-2-CTD −0.322)
shows the CTD embedding is *not* more discriminative even though Li 2024 names the CTD as
the specificity determinant — because the CTD acts through **conformation**, which a
sequence embedding (and AlphaFold3) cannot read. This confirms the barrier is intrinsic
(4 examples + a conformational, non-sequence-readable determinant), not feature quality,
and is why the evidence must come from biology (below) rather than an internal score.

**Tier 2/3 — do the C. briggsae predictions hold up biologically? (`tier23_evaluation.py`).**
Because C. elegans motifs do not occur in C. briggsae, we evaluate the *model-free*
terminal k-mer enrichment (`cb_kmer_enrichment.csv`) — the unsupervised candidate PC
motifs — per chromosome, checking end-clustering, chromosome-specificity, tandem genomic
spacing, the TTGG-spacer-TG framework, and convergence with the C. elegans paralog known
to bind each chromosome. Output: `cb_pc_candidates.csv`, `cb_pc_candidates.png`.

**Headline: 4 of 6 C. briggsae chromosomes carry a chromosome-specific, TTGG-TG-framework
terminal motif matching the expected paralog** (permutation null: 1.85/6 expected by
chance; **P(≥4) = 0.068** — suggestive, not conclusive at 0.05). The credible, tandem
cases are the strong ones:

| Chr | Candidate | Spacer | Specificity | Tandem spacing | Matches | Note |
|-----|-----------|:------:|:-----------:|:--------------:|---------|------|
| III | `TTGGCCTGCTAA` | 2 | 0.81 | 1298 bp | **ZIM-1** ✓ | best match (skill 0.67); spacer 2 = ZIM-1's spacer |
| X   | `TTGGTAGTGGTT` | 3 | 1.00 | 140 bp | **HIM-8** ✓ | X is the HIM-8 target; tight tandem |
| IV  | `TTGGGGCATGAC` | 4 | 1.00 | 88 bp | **ZIM-3** ✓ | tight tandem, chromosome-specific |
| I   | `TTGGATTTGTCT` | 3 | 1.00 | 28.5 kb | ZIM-3 ✓ | weak (not tandem) |
| II  | `TTGGCCTTGTGG` | 3 | 1.00 | 3473 bp | ZIM-3 | non-convergent (expected ZIM-1) |
| V   | `TTGGTACTGCCT` | 3 | 1.00 | 4466 bp | ZIM-1 | non-convergent (expected ZIM-2) |

Caveats (stated honestly): p = 0.068 is suggestive only; ZIM-3 is a frequent best-match
(a possible loose-match bias for its short-spacer motif); chr I's convergent call is weak
(not tandem); and the low-complexity micro-repeat `TTGGCTTGGCTT` (5-bp spacing on chr I/II)
was flagged and excluded. The robust signal is chr **III (ZIM-1)**, **X (HIM-8)**, and **IV
(ZIM-3)** — chromosome-specific, tandemly-arranged, framework-following motifs matching the
expected paralog, from an analysis that uses no protein model at all. This is independent,
convergent support for the project premise and the ortholog/chromosome assignments.

---

## Layer 6 — De novo motif discovery from the genome (`discover_motifs.py`)

The reframing that actually works. Protein→motif prediction is unsolvable here (4 examples;
a conformational determinant sequence models can't read). But the genome IS sequenced, and a
pairing center is a **dense tandem array of a short motif near a chromosome end** — so we
find the motif *directly from the genome*, where thousands of copies sit, instead of
predicting it from the protein. Per chromosome: seed with the top enriched TTGG-framework
k-mer → localize the terminal cluster (peak of near-seed TTGG sites in a 50 kb window) →
EM-refine a PWM **constrained to keep the TTGG-spacer-TG framework** (otherwise it drifts to
the AT-rich / micro-repeat sequences that also pile up at chromosome ends) → assign to a
paralog by synteny + locked-metric similarity to the C. elegans motif. Output:
`discovered_motifs.csv/npz`, and sequence logos in `discovered_motifs.png`.

**Cross-assembly validation.** Discovery was run on **both** the AF16 (PRJNA10731) and the
chromosome-level **QX1410 (PRJNA784955)** assemblies — two independently-sequenced strains —
and reconciled (`discover_motifs.py --reconcile` → `discovered_motifs_final.csv`, `_final.png`).
A motif that reproduces across both strains is far stronger than any single run. The pipeline
searches both chromosome ends, reports the cluster's genomic coordinates and distance-from-end,
scores chromosome-specificity, and assigns the binding protein by **synteny** (chromosome →
paralog, the reliable basis for diverged motifs), with motif similarity as an independent
cross-check. Protein-per-genome caveat: pairing centers sit at chromosome ends, so QX1410's
clean ends matter; end N-content was < 0.05 on both assemblies here.

**All six motifs reproduced across both assemblies (6/6); four are HIGH-confidence.** Final
predictions (`discovered_motifs_final.csv`):

| Chr | Binds (synteny) | Motif | Spacer | Copies | Period | Down-bits | Motif~ | Confidence |
|-----|-----------------|-------|:------:|:------:|:------:|:---------:|--------|:----------:|
| III | **Cbr-ZIM-1** | `TTGGTCTGCTAATTAT` | 2 | 102 | 44 bp | 15.5 | ZIM-1 (skill 0.86) ✓ | **HIGH** |
| IV  | **Cbr-ZIM-3** | `TTGGGTCATGACCTAG` | 4 | 113 | 22 bp | 19.2 | ZIM-3 ✓ | **HIGH** |
| X   | **Cbr-HIM-8** | `TTGGTAGTGGTTCCGC` | 3 | 39 | 140 bp | 18.2 | HIM-8 ✓ | **HIGH** |
| V   | Cbr-ZIM-2 | `TTGGTACTGCTTAGCA` | 3 | 47 | 163 bp | 14.9 | ZIM-1 (disagrees) | HIGH* |
| II  | Cbr-ZIM-1 | `TTGGTCTTGTGGACCA` | 3 | 34 | 931 bp | 8.5 | ZIM-3 (disagrees) | MEDIUM |
| I   | Cbr-ZIM-3 | `TTGGGTTGGAAGAAAT` | 2 | 103 | 288 bp | 2.3 | ZIM-3 ✓ | LOW |

**Gold-standard — 3 predictions where confidence is HIGH _and_ synteny agrees with motif
similarity:** **Cbr-ZIM-1 → `TTGGTCTGCTAATTAT` (chr III)**, **Cbr-ZIM-3 → `TTGGGTCATGACCTAG`
(chr IV)**, and **Cbr-HIM-8 → `TTGGTAGTGGTTCCGC` (chr X)** — the last a **previously
uncharacterised candidate for the C. briggsae X-chromosome pairing center**, exactly the
prediction the project set out to make, obtained by discovery from the genome rather than a
protein model. chr V's motif is well-defined and reproduced, but its protein is ambiguous
(synteny says ZIM-2, motif similarity says ZIM-1 — flagged for experimental confirmation);
chr I/II are weaker.

Caveats (honest): no ground truth — these are candidates to confirm by ChIP / experiment
(route via the professor); the protein assignment rests on chromosome→paralog synteny, robust
for the three gold-standard cases (both lines of evidence agree) but ambiguous for chr V/II.

---

## Reproducing

From the project root, with the venv active and raw data in `data/raw/`:

```bash
python eda.py                  # ~15 min (genome-wide k-mer scan)
python build_selex_pwms.py     # seconds
python find_orthologs.py       # ~2-3 min
python find_orthologs_blast.py # ~1 min (needs BLAST+; brew install blast)
python build_gene_tree.py      # ~1 min (needs MAFFT;  brew install mafft)
python compute_embeddings.py   # frozen ESM-2 650M domain embeddings (one-time)
python run_baselines.py        # Baselines 1-2 + C. briggsae predictions (seconds)
python pretrain_jaspar.py      # JASPAR download + embed + PLS (~20 min first run, then cached)
python train_head.py           # Baseline 3 learned head + L2 sweep (seconds)
python recognition_baseline.py # biology-informed ZF recognition-helix features (~1 min)
python tier23_evaluation.py    # Tier 2/3 C. briggsae biology (~1-2 min; needs the genome)
python discover_motifs.py <qx1410.fa.gz> qx1410   # discovery on the chromosome-level assembly
python discover_motifs.py <af16.fa.gz>   af16     # discovery on AF16 (cross-check)
python discover_motifs.py --reconcile             # final cross-assembly prediction sheet
```

The first five scripts are deterministic and reproduce the numbers in this document.
`pretrain_jaspar.py` downloads from JASPAR/UniProt on first run and caches under
`data/raw/jaspar/` and `data/processed/jaspar_embeddings.npz`; given the cached
projection, the learned-head scores are deterministic.
