"""Decide which downloaded species are fit to train on.

The instruction for this dataset is explicit: several of the deposited species do
not have finished genome annotations, and those must not be used. This script is
where that decision is made once, written down, and made auditable — every later
step reads the table it produces rather than re-deciding for itself.

Two modes:

    python prepare_species.py --scan     draft a manifest from whatever is sitting
                                         in data/raw/ (run this first, then check it)
    python prepare_species.py            assess the manifest and write species_qc.csv

A species is used only if the data source lists its annotation as complete *and*
the files actually contain a complete annotation (a real proteome, a real gene set,
chromosome-scale sequences with clean ends). Anything that fails is reported with
the reason, never dropped in silence.
"""
import os
import sys

import pandas as pd

from src.paths import P
from src.species import (MANIFEST, RAW_ROOT, Species, assess, load_manifest)

# Filename hints used when drafting a manifest from a directory of downloads.
GENOME_HINTS = ('genomic', 'genome', 'dna', 'softmasked', 'toplevel')
PROTEIN_HINTS = ('protein', 'pep', 'aa', 'proteins')
ANNOTATION_EXT = ('.gff3', '.gff3.gz', '.gff', '.gff.gz', '.gtf', '.gtf.gz')
SEQ_EXT = ('.fa', '.fa.gz', '.fasta', '.fasta.gz', '.fna', '.fna.gz')


def _classify(files):
    """Sort one species folder's files into genome / proteins / annotation.

    Matching is by the conventions the genome repositories actually use: the word
    'protein' or 'pep' in a FASTA name means a proteome, 'genomic'/'dna' means the
    assembly, and a GFF3/GTF is the annotation. Proteins are checked first because
    a protein file is also a FASTA and would otherwise be taken for the assembly.
    """
    genome = proteins = annotation = ''
    for f in sorted(files):
        low = os.path.basename(f).lower()
        if low.endswith(ANNOTATION_EXT):
            annotation = annotation or f
        elif low.endswith(SEQ_EXT):
            if any(h in low for h in PROTEIN_HINTS):
                proteins = proteins or f
            elif any(h in low for h in GENOME_HINTS):
                genome = genome or f
            else:
                genome = genome or f          # an unlabelled FASTA is the assembly
    return genome, proteins, annotation


def _pretty_name(key):
    """'c_briggsae_qx1410' -> 'Caenorhabditis briggsae qx1410'."""
    parts = key.replace('-', '_').split('_')
    if parts and parts[0] in ('c', 'cae'):
        parts[0] = 'Caenorhabditis'
    return ' '.join([parts[0].capitalize()] + parts[1:]) if parts else key


def _short(key, used=()):
    """'c_briggsae' -> 'Cbr' — the prefix that names this species' proteins.

    Must be unique across the manifest: the prefix plus the paralog is the key a
    protein is looked up by, so two species sharing a prefix would silently overwrite
    each other's embeddings. Two assemblies of the same species collide by
    construction (c_briggsae_af16 and c_briggsae_qx1410 both give 'Cbr'), so a
    colliding prefix is extended with the strain, then with a digit.
    """
    parts = [p for p in key.replace('-', '_').split('_') if p]
    base = ((parts[0][0] + parts[1][:2]).capitalize() if len(parts) >= 2
            else key[:3].capitalize())
    if base not in used:
        return base
    if len(parts) >= 3:                       # distinguish by strain: 'Cbr' -> 'Cbrqx'
        strain = base + parts[2][:2].lower()
        if strain not in used:
            return strain
    for i in range(2, 100):
        if f'{base}{i}' not in used:
            return f'{base}{i}'
    return base


def scan(root=RAW_ROOT, out=MANIFEST):
    """Draft a manifest by looking at what was actually downloaded.

    Each immediate subdirectory of data/raw/ is treated as one species. A species
    whose folder has no proteome or no GFF3 is drafted as annotation-incomplete,
    because that is precisely what an unfinished annotation looks like on disk:
    the assembly is deposited but the gene set is not.

    The draft is a starting point, not the last word — open it and correct the
    `annotation_complete` column against the data source's own listing before
    training on anything.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No {root}/ directory to scan. Download the genomes "
                                f"there first, one folder per species.")
    rows, used_shorts = [], set()
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder) or entry.startswith('.'):
            continue
        files = []
        for dirpath, _, filenames in os.walk(folder):
            files += [os.path.join(dirpath, f) for f in filenames]
        genome, proteins, annotation = _classify(files)
        if not genome:
            continue                       # not a species folder

        missing = [n for n, v in (('proteome', proteins), ('annotation', annotation))
                   if not v]
        short = _short(entry, used_shorts)
        used_shorts.add(short)
        rows.append({
            'key': entry,
            'name': _pretty_name(entry),
            'short': short,
            # Paths are stored relative to data/raw/ so the manifest stays readable.
            'genome': os.path.relpath(genome, root),
            'proteins': os.path.relpath(proteins, root) if proteins else '',
            'annotation': os.path.relpath(annotation, root) if annotation else '',
            'annotation_complete': 'no' if missing else 'yes',
            'notes': (f"drafted by --scan; no {' or '.join(missing)} file found"
                      if missing else 'drafted by --scan; verify against source listing'),
            'chrom_map': '',
        })

    if not rows:
        raise SystemExit(f"No species folders with a genome FASTA found under {root}/.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Drafted {out} with {len(df)} species:\n")
    print(df[['key', 'annotation_complete', 'genome', 'proteins', 'annotation']]
          .to_string(index=False))
    print(f"\nCheck the annotation_complete column against the data source's own "
          f"listing of which annotations are finished, then run:\n"
          f"  python prepare_species.py")
    return df


def main():
    if '--scan' in sys.argv:
        scan()
        return

    species = load_manifest()
    print(f"Assessing {len(species)} species from {MANIFEST}...\n")

    reports = []
    for s in species:
        print(f"  {s.key:28s} ...", end=' ', flush=True)
        rep = assess(s)
        reports.append(rep)
        verdict = ('labels+proteins' if rep['usable_for_labels']
                   else 'proteins only' if rep['usable_for_proteins'] else 'EXCLUDED')
        print(verdict)

    df = pd.DataFrame(reports)
    df.to_csv(P('species_qc.csv'), index=False)

    pd.set_option('display.width', 220, 'display.max_columns', 20)
    cols = ['key', 'declared_complete', 'n_proteins', 'n_genes', 'n_chromosomes',
            'assembly_span_mb', 'worst_end_n', 'usable_for_proteins',
            'usable_for_labels']
    print('\n' + df[cols].to_string(index=False))

    excluded = df[~df['usable_for_labels']]
    if len(excluded):
        print("\nNot contributing training labels:")
        for _, r in excluded.iterrows():
            print(f"  {r['key']:28s} {r['reasons']}")

    n_labels = int(df['usable_for_labels'].sum())
    print(f"\n{n_labels} of {len(df)} species will contribute training labels.")
    if n_labels < 3:
        print("  Warning: leave-one-species-out needs at least 3 usable species.")
    print(f"Saved {P('species_qc.csv')}")


if __name__ == '__main__':
    main()
