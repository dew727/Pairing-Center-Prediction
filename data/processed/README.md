# `data/processed/` — generated results, organized by method / run stage

Every file here is produced by a script (routed automatically via `src/paths.py`).
Folders are ordered by pipeline stage.

| Folder | What's in it | Produced by | Key result |
|--------|--------------|-------------|------------|
| `01_eda/` | genome stats, motif-scan density plots, motif hit tables, terminal k-mer enrichment | `eda.py` | `cb_kmer_enrichment.csv` |
| `02_selex_pwms/` | SELEX-derived PWMs + provenance | `build_selex_pwms.py` | `pwms.npz` |
| `03_orthologs/` | C. briggsae ortholog tables (Python + BLAST) and the ZIM/HIM-8 gene tree | `find_orthologs.py`, `find_orthologs_blast.py`, `build_gene_tree.py` | `ortholog_table.csv` |
| `04_embeddings/` | frozen ESM-2 embeddings (full, domain, recognition-helix) + metadata | `compute_embeddings.py`, `compare_domain_embeddings.py`, `recognition_baseline.py` | `embeddings_domain.npz` |
| `05_baselines/` | Baseline 1/2/3 leave-one-paralog-out scores + C. briggsae nearest-neighbour predictions | `run_baselines.py`, `train_head.py` | `head_scores.csv` |
| `06_jaspar_pretraining/` | JASPAR C2H2 embeddings + PLS projection for Baseline 3 pretraining | `pretrain_jaspar.py` | `jaspar_projection.npz` |
| `07_tier23_biology/` | Tier 2/3 biological-plausibility evaluation of the C. briggsae predictions | `tier23_evaluation.py` | `cb_pc_candidates.csv` |
| `08_discovery_ttgg/` | de novo PC-motif discovery with the TTGG framework (AF16, QX1410, reconciled final) | `discover_motifs.py` | `discovered_motifs_final.csv` |
| `09_discovery_anchorfree/` | drift-safe control: anchor-free, specificity-selected discovery (C. briggsae, C. remanei) | `discover_denovo.py` | `denovo_motifs_qx1410.csv` |
| `10_anchored_reconciliation/` | drift-aware sheet: TTGG vs anchor-free motif per chromosome + recognition-residue conservation | `discover_anchored.py` | `anchored_predictions.csv` |
| `11_cremanei/` | C. remanei cross-species replication (TTGG run + run comparison) | `crem_ttgg_check.py`, `compare_runs.py` | `crem_run_comparison.csv` |

Raw genomes/annotations are **not** in git (large; see `RESULTS.md` for WormBase accessions).
All paths are defined centrally in `src/paths.py`, so re-running any script writes back into
the correct subfolder here.
