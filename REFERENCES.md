# References

Reading list for the methods and biology used in this project, organized by
step. Citations are given by author / year / title / journal so they can be
found on PubMed or Google Scholar; verify exact volume/page before citing
formally. Domain papers marked † are the project's primary sources.

## The biological problem (pairing centers, ZIM/HIM-8)
- † **Li et al. 2024**, *Nat Commun* — ZIM/HIM-8 crystal structures; the conserved
  TTGG / TG sub-sites, the variable spacer between paralogs, and the documented
  AlphaFold-3 failure that motivates a specialized sequence→motif model.
- † **Phillips et al. 2009**, *Nat Cell Biol* (DOI 10.1038/ncb1904) — the SELEX
  bound-oligo data converted into PWMs here.
- † **Phillips & Dernburg 2006**, *Dev Cell* — "A family of zinc-finger proteins…";
  defines the paralog family and the C. briggsae orthologs (the "5 paralogs").
- **Phillips et al. 2005**, *Cell* — original "HIM-8 binds the X pairing center";
  background on what these proteins do during meiosis.
- † **Rillo-Bohn et al. 2021** — phylogenetic scope of the family across nematodes.

## Motifs, PWMs, SELEX (build_selex_pwms.py)
- **Stormo 2000**, *Bioinformatics*, "DNA binding sites: representation and
  discovery" — canonical intro to PWMs, pseudocounts, information content.
- **Schneider & Stephens 1990**, *Nucleic Acids Res*, "Sequence logos" — reading a
  PWM as a logo; relevant to interpreting consensus columns.
- **Stormo & Zhao 2010**, *Nat Rev Genet*, "Determining the specificity of
  protein–DNA interactions" — review linking binding data to specificity models.
- **Tuerk & Gold 1990**, *Science* — the original SELEX method.

## Homology search & ortholog detection (find_orthologs*.py)
- **Altschul et al. 1990**, *J Mol Biol*, "Basic local alignment search tool" — BLAST.
- **Altschul et al. 1997**, *Nucleic Acids Res*, "Gapped BLAST and PSI-BLAST" —
  gapped/iterated BLAST and E-value statistics.
- **Henikoff & Henikoff 1992**, *PNAS*, "Amino acid substitution matrices from
  protein blocks" — BLOSUM62 (used by BLAST and the Python search).
- **Tatusov, Koonin & Lipman 1997**, *Science* — reciprocal-best-hit / COGs, the
  basis of the RBH check.

## Orthology vs paralogy & co-orthology (build_gene_tree.py)
- **Sonnhammer & Koonin 2002**, *Trends Genet*, "Orthology, paralogy and proposed
  classification for paralog subtypes" — defines **co-orthologs** and
  **in-paralogs**, the exact terms for the zim subfamily result. Read closely.
- **Fitch 1970**, *Syst Zool*, "Distinguishing homologous from analogous proteins"
  — origin of the ortholog/paralog distinction (speciation vs duplication node).
- **Koonin 2005**, *Annu Rev Genet*, "Orthologs, paralogs, and evolutionary
  genomics" — review incl. lineage-specific expansions.
- **Maddison 1997**, *Syst Biol*, "Gene trees in species trees" — why a gene tree
  can disagree with the species tree (licenses the duplication-vs-speciation read).

## Phylogenetics methods (MAFFT + neighbour-joining + bootstrap)
- **Saitou & Nei 1987**, *Mol Biol Evol*, "The neighbor-joining method".
- **Felsenstein 1985**, *Evolution*, "Confidence limits on phylogenies… the
  bootstrap" — what the bootstrap support numbers mean.
- **Katoh & Standley 2013**, *Mol Biol Evol*, "MAFFT… version 7" — the aligner used.
- **Felsenstein 1978**, *Syst Zool*, "Cases in which parsimony or compatibility
  methods will be positively misleading" — long-branch attraction (the caveat on
  the species-clustering topology).
- Textbook: **Felsenstein, *Inferring Phylogenies* (2004)**.

## Planned modeling (ESM-2 → motif)
- **Lin et al. 2023**, *Science*, "Evolutionary-scale prediction of atomic-level
  protein structure with a language model" — ESM-2 / ESMFold (the feature extractor).
- **Rives et al. 2021**, *PNAS* — earlier ESM; why frozen embeddings carry
  structural/functional signal (justifies not fine-tuning on 4 examples).
- **Abramson et al. 2024**, *Nature*, "AlphaFold 3" — to ground the failure mode
  being worked around.
- **Persikov & Singh 2014**, *Nucleic Acids Res*, "De novo prediction of
  DNA-binding specificities for Cys2His2 zinc finger proteins" — closest prior art.
- **Alipanahi et al. 2015**, *Nat Biotechnol*, "DeepBind" — deep-learning binding
  specificity prediction; baseline-philosophy reference.
