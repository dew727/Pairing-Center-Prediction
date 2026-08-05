"""The registry of species used for training, and the annotation-completeness gate.

The first model was trained on four proteins, all from *C. elegans*, and that is
why it failed: four examples cannot teach a model which letters a fifth protein
prefers. The fix is to bring in more species. But a genome is only useful here if
its annotation is finished — we need the *proteome* (to find the ZIM/HIM-8
orthologs) and a *chromosome-level assembly* (because a pairing center sits at a
chromosome end, so a genome shattered into scaffolds has no "end" to look at).

Some of the deposited species are flagged by the data provider as not having a
complete annotation. Those must not be trained on. This module encodes that in two
independent layers, and a species has to pass both:

  1. The *declared* status, read from `data/species_manifest.csv` — the provider's
     own listing of which annotations are complete.
  2. A *verified* status, computed here by actually opening the files and checking
     they contain what a complete annotation should contain.

Layer 2 exists because a declared status is a claim about the data, not the data
itself. If a species is listed as complete but its protein file holds 40 sequences,
we exclude it and say so, rather than quietly training on a stub.
"""
import json
import os
from dataclasses import dataclass, field

from src.data_io import read_fasta, read_gff

# The six Caenorhabditis chromosomes. Everything else in an assembly (mitochondrion,
# unplaced scaffolds) is ignored: pairing centers live on these six.
MAIN_CHROMS = ['I', 'II', 'III', 'IV', 'V', 'X']

# What a finished Caenorhabditis annotation should look like. These are deliberately
# loose — they are meant to catch stubs and assembly-only deposits, not to second-guess
# a genuine annotation that happens to be a little smaller than its relatives.
MIN_PROTEINS = 5_000          # a real proteome is ~20k; a placeholder is far smaller
MIN_GENES = 5_000             # same reasoning, counted from the GFF3
MIN_CHROM_LENGTH = 5_000_000  # a real Caenorhabditis chromosome is 13-21 Mb
MIN_CHROMS_FOUND = 5          # allow one chromosome to be missing or misnamed
MAX_END_N_FRACTION = 0.05     # gappy ends make a terminal motif search unreliable

MANIFEST = 'data/species_manifest.csv'
RAW_ROOT = 'data/raw'


@dataclass
class Species:
    """One species' files and its place in the study.

    `declared_complete` is what the data provider says; `notes` carries their wording
    verbatim so an exclusion can always be traced back to its source.
    """
    key: str                      # short unique id, e.g. 'c_briggsae_qx1410'
    name: str                     # 'Caenorhabditis briggsae'
    short: str                    # 'Cbr' — the prefix used in embedding keys
    genome: str                   # genomic FASTA (.fa / .fa.gz)
    proteins: str                 # protein FASTA
    annotation: str               # GFF3
    declared_complete: bool = True
    notes: str = ''
    chrom_map: dict = field(default_factory=dict)   # assembly seq name -> 'I'..'X'

    def paths(self):
        return {'genome': self.genome, 'proteins': self.proteins,
                'annotation': self.annotation}

    def missing_files(self):
        return [k for k, p in self.paths().items() if not p or not os.path.exists(p)]


def normalize_chrom(name, chrom_map=None):
    """Map an assembly's sequence name onto one of I, II, III, IV, V, X.

    Assemblies name their chromosomes inconsistently ('I', 'chrI', 'CM008945.1',
    'chr_I'). A per-species `chrom_map` in the manifest handles accessions that
    cannot be guessed; otherwise we strip the common prefixes and check the result
    against the six real names. Returns None for anything else (scaffolds, the
    mitochondrion), which is how those get dropped.
    """
    if chrom_map and name in chrom_map:
        return chrom_map[name]
    n = name.strip()
    for prefix in ('chromosome_', 'chromosome', 'chrom_', 'chr_', 'chr'):
        if n.lower().startswith(prefix):
            n = n[len(prefix):]
            break
    n = n.strip('_-').upper()
    return n if n in MAIN_CHROMS else None


def load_genome(species):
    """Read a genome and return {normalized chromosome name: uppercase sequence}.

    Only the six real chromosomes are kept, under their standard names, so every
    downstream step can assume the same naming regardless of the assembly.
    """
    genome = {}
    for name, seq in read_fasta(species.genome).items():
        chrom = normalize_chrom(name, species.chrom_map)
        if chrom is not None:
            genome[chrom] = seq.upper()
    return genome


def _end_n_fraction(seq, window=500_000):
    """Worst N-content across the two chromosome ends.

    Pairing centers sit at the ends, so a genome with gappy ends cannot support a
    terminal motif search even if the rest of the assembly is fine.
    """
    w = min(window, len(seq) // 2) or 1
    return max(seq[:w].count('N') / w, seq[-w:].count('N') / w)


def assess(species):
    """Open a species' files and judge whether its annotation is actually complete.

    Returns a report with the measured numbers, a verdict, and — when a species is
    rejected — the specific reasons. Two verdicts matter downstream:

      `usable_for_labels`   the species can contribute training motifs (needs a
                            chromosome-level assembly with clean ends, because the
                            motif is read off the chromosome ends).
      `usable_for_proteins` the species' ZIM/HIM-8 proteins can be used at all.

    A species that fails is never silently dropped; the reasons travel with it into
    `species_qc.csv`.
    """
    report = {'key': species.key, 'name': species.name, 'short': species.short,
              'declared_complete': species.declared_complete, 'notes': species.notes}
    reasons = []

    missing = species.missing_files()
    if missing:
        report.update(n_proteins=0, n_genes=0, n_chromosomes=0,
                      assembly_span_mb=0.0, worst_end_n=None,
                      usable_for_proteins=False, usable_for_labels=False,
                      reasons='missing files: ' + ','.join(missing))
        return report

    # The provider's own listing is authoritative for exclusion: if they say the
    # annotation is incomplete, we do not try to talk ourselves out of it.
    if not species.declared_complete:
        reasons.append('declared incomplete by data source')

    proteins = read_fasta(species.proteins)
    n_proteins = len(proteins)
    if n_proteins < MIN_PROTEINS:
        reasons.append(f'only {n_proteins} proteins (< {MIN_PROTEINS})')

    gff = read_gff(species.annotation)
    n_genes = int((gff['type'] == 'gene').sum())
    if n_genes < MIN_GENES:       # some GFFs label the feature 'mRNA' instead
        n_genes = max(n_genes, int((gff['type'] == 'mRNA').sum()))
    if n_genes < MIN_GENES:
        reasons.append(f'only {n_genes} annotated genes (< {MIN_GENES})')

    genome = load_genome(species)
    full = {c: s for c, s in genome.items() if len(s) >= MIN_CHROM_LENGTH}
    n_chroms = len(full)
    span_mb = round(sum(len(s) for s in full.values()) / 1e6, 1)
    if n_chroms < MIN_CHROMS_FOUND:
        reasons.append(f'only {n_chroms} chromosome-scale sequences '
                       f'(< {MIN_CHROMS_FOUND}); assembly is not chromosome-level')

    worst_end_n = (round(max(_end_n_fraction(s) for s in full.values()), 3)
                   if full else None)
    if worst_end_n is not None and worst_end_n > MAX_END_N_FRACTION:
        reasons.append(f'gappy chromosome ends (N={worst_end_n:.3f} '
                       f'> {MAX_END_N_FRACTION})')

    # Proteins are usable whenever the proteome itself is complete; motif labels
    # additionally require real chromosomes with clean ends.
    protein_ok = (species.declared_complete and n_proteins >= MIN_PROTEINS
                  and n_genes >= MIN_GENES)
    label_ok = (protein_ok and n_chroms >= MIN_CHROMS_FOUND
                and worst_end_n is not None and worst_end_n <= MAX_END_N_FRACTION)

    report.update(n_proteins=n_proteins, n_genes=n_genes, n_chromosomes=n_chroms,
                  assembly_span_mb=span_mb, worst_end_n=worst_end_n,
                  usable_for_proteins=protein_ok, usable_for_labels=label_ok,
                  reasons='; '.join(reasons))
    return report


def _parse_bool(value, default=True):
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ('', 'nan'):
        return default
    if s in ('1', 'true', 'yes', 'y', 'complete', 'completed'):
        return True
    if s in ('0', 'false', 'no', 'n', 'incomplete', 'partial', 'draft'):
        return False
    return default


def load_manifest(path=MANIFEST):
    """Read the species manifest into Species objects.

    The manifest is the written-down record of what was downloaded and which
    annotations the provider listed as complete. Relative file paths are resolved
    against `data/raw/`, so the manifest stays readable.
    """
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No species manifest at {path}. Create one (see data/species_manifest.csv "
            f"in the repo for the expected columns) listing each downloaded species "
            f"and whether its annotation is complete.")

    df = pd.read_csv(path, dtype=str).fillna('')
    species = []
    for _, r in df.iterrows():
        if not r.get('key'):
            continue
        chrom_map = json.loads(r['chrom_map']) if r.get('chrom_map') else {}

        def resolve(col):
            v = r.get(col, '')
            if not v:
                return ''
            return v if os.path.isabs(v) else os.path.join(RAW_ROOT, v)

        species.append(Species(
            key=r['key'], name=r.get('name', r['key']), short=r.get('short', r['key'][:3]),
            genome=resolve('genome'), proteins=resolve('proteins'),
            annotation=resolve('annotation'),
            declared_complete=_parse_bool(r.get('annotation_complete')),
            notes=r.get('notes', ''), chrom_map=chrom_map,
        ))

    # The short prefix plus the paralog is how a protein is looked up, so two species
    # sharing a prefix would overwrite each other's embeddings and mislabel the
    # training set. Two assemblies of one species collide easily, so check explicitly.
    for field_name in ('key', 'short'):
        seen = {}
        for s in species:
            value = getattr(s, field_name)
            if value in seen:
                raise ValueError(
                    f"Duplicate {field_name} {value!r} in {path}: used by both "
                    f"{seen[value]!r} and {s.key!r}. Each species needs a unique "
                    f"{field_name}, because it identifies that species' proteins.")
            seen[value] = s.key
    return species
