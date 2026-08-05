"""Build the training set: one (protein, motif) pair per chromosome, per species.

This is the step that fixes the original problem. The first model had four labelled
proteins, all *C. elegans*, and no amount of modelling can make four hyper-specific
examples generalise. Here every additional species with a finished annotation
contributes its own ZIM/HIM-8 proteins paired with the motifs read off its own
chromosome ends, so the training set grows with the number of genomes instead of
being fixed at four.

For each species that passed the annotation gate:

    1. find the ZIM/HIM-8 orthologs in its proteome
    2. discover the pairing-center motif at each chromosome end from its genome
    3. join the two by synteny (which paralog binds which chromosome)

*C. elegans* is handled differently and deliberately: its motifs are known from the
lab, so running the same discovery on it is a **control**. How close the discovered
C. elegans motifs come to the measured ones is a direct measurement of how good the
silver labels are, and it is reported here. Those control rows are not trained on —
the measured C. elegans motifs are held back as the test set.

Run from the project root, after prepare_species.py:
    python build_multispecies_labels.py
Writes data/processed/12_multispecies/silver_labels.csv (+ .npz of the PWMs).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src.paths import P
from src.data_io import read_fasta
from src.species import load_manifest, load_genome
from src.silver import (PARALOGS, build_labels, discover_species, find_orthologs_in,
                        species_seeds, summarize)
from src.motifs import KNOWN_MOTIFS
from find_orthologs import CE_PROTEINS, QUERY_LOCI, read_headers, representative_queries
from src.metrics import PARALOG_SPACER
from src.pcmodel import score_native
from src.motifs import anchored_transfer, motif_to_pwm

QC_TABLE = P('species_qc.csv')
CACHE_DIR = os.path.join('data', 'processed', '12_multispecies', 'cache')


def elegans_queries(manifest):
    """The four C. elegans proteins used to find orthologs in every other species.

    Preferentially read from whichever manifest entry is C. elegans, falling back to
    the path the single-species pipeline already used.
    """
    path = CE_PROTEINS
    for s in manifest:
        if 'elegans' in s.key.lower() and s.proteins and os.path.exists(s.proteins):
            path = s.proteins
            break
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"C. elegans proteome not found at {path}; it is needed as the query set "
            f"for finding ZIM/HIM-8 orthologs in every other species.")
    proteins = read_fasta(path)
    headers = read_headers(path)
    chosen = representative_queries(proteins, headers, QUERY_LOCI)
    # 'zim-1' in the FASTA becomes 'ZIM-1' as the metric names it.
    return {locus.upper(): seq for locus, (pid, seq, _) in chosen.items()}


def cached_discovery(species, genome, refresh=False):
    """Discover this species' motifs, reusing a cached result when there is one.

    Discovery counts every k-mer in the genome, which takes minutes per species, so
    the result is cached; re-running the pipeline to adjust the model should not
    re-read the genomes.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta_path = os.path.join(CACHE_DIR, f'{species.key}_discovery.json')
    pwm_path = os.path.join(CACHE_DIR, f'{species.key}_discovery.npz')

    if not refresh and os.path.exists(meta_path) and os.path.exists(pwm_path):
        with open(meta_path) as f:
            records = json.load(f)
        pwms = np.load(pwm_path)
        for r in records:
            r['pwm'] = pwms[r['chromosome']]
        print(f"    (cached) {len(records)} chromosome motifs")
        return records

    t0 = time.time()
    seeds = species_seeds(genome)
    print(f"    seeds on {len(seeds)} chromosomes; discovering...", flush=True)
    records = discover_species(genome, seeds)

    np.savez(pwm_path, **{r['chromosome']: r['pwm'] for r in records})
    with open(meta_path, 'w') as f:
        json.dump([{k: (v if not isinstance(v, np.generic) else v.item())
                    for k, v in r.items() if k != 'pwm'} for r in records], f, indent=2)
    print(f"    {len(records)} chromosome motifs in {time.time() - t0:.0f}s")
    return records


def control_quality(records):
    """How well discovery recovers the *known* C. elegans motifs.

    This is the honest measure of silver-label quality. Each discovered C. elegans
    motif is scored against the measured motif of the paralog that binds that
    chromosome, using the project's locked metric. A high score means the same
    procedure applied to other species is producing trustworthy labels; a low score
    means the silver labels are noise and the whole approach should be distrusted.
    """
    from src.silver import CE_CHROM_PARALOG
    rows = []
    for r in records:
        paralog = CE_CHROM_PARALOG.get(r['chromosome'])
        if paralog is None:
            continue
        true = KNOWN_MOTIFS[paralog]
        # Re-express the discovered motif in the known motif's frame, registered on
        # the shared TTGG and TG anchors. The two can have different spacer lengths,
        # so lining them up from the left would compare unrelated positions.
        pred = anchored_transfer(r['pwm'][:, :12], r['spacer'],
                                 len(true), PARALOG_SPACER[paralog])
        res = score_native(pred, motif_to_pwm(true), PARALOG_SPACER[paralog])
        rows.append({'chromosome': r['chromosome'], 'paralog': paralog,
                     'discovered': r['motif'], 'known': true,
                     'confidence': r['confidence'],
                     'skill_vs_known': round(res['skill'], 3)})
    return rows


def main():
    refresh = '--refresh' in sys.argv
    manifest = load_manifest()
    by_key = {s.key: s for s in manifest}

    if not os.path.exists(QC_TABLE):
        raise SystemExit(f"No {QC_TABLE}. Run `python prepare_species.py` first — it "
                         f"decides which annotations are complete enough to use.")
    qc = pd.read_csv(QC_TABLE)
    usable = qc[qc['usable_for_labels']]
    print(f"{len(usable)} of {len(qc)} species passed the annotation gate.\n")
    if usable.empty:
        raise SystemExit("No species with a complete annotation; nothing to build.")

    queries = elegans_queries(manifest)
    missing_q = [p for p in PARALOGS if p not in queries]
    if missing_q:
        raise SystemExit(f"Missing C. elegans query proteins: {missing_q}")

    all_labels, all_pwms, control_rows, ortholog_rows = [], {}, [], []

    for _, row in usable.iterrows():
        species = by_key[row['key']]
        is_elegans = 'elegans' in species.key.lower()
        tag = ' [control: motifs are known]' if is_elegans else ''
        print(f"  {species.key}{tag}")

        genome = load_genome(species)
        proteome = read_fasta(species.proteins)
        print(f"    {len(genome)} chromosomes, {len(proteome)} proteins")

        orthologs = find_orthologs_in(proteome, queries)
        for paralog, o in orthologs.items():
            ortholog_rows.append({'species': species.key, 'paralog': paralog, **o})
        print(f"    orthologs: " + ', '.join(
            f"{p}={orthologs[p]['protein_id']}({orthologs[p]['identity']}%)"
            for p in PARALOGS if p in orthologs))

        records = cached_discovery(species, genome, refresh=refresh)
        if is_elegans:
            control_rows += control_quality(records)

        labels = build_labels(species.key, species.name, species.short,
                              orthologs, records)
        # C. elegans discoveries are a control only; its measured motifs are the test set.
        for lab in labels:
            if is_elegans:
                lab['label_type'] = 'control'
                lab['used_for_training'] = False
        all_labels += labels
        for r in records:
            all_pwms[f"{species.key}:{r['chromosome']}"] = r['pwm']
        print()

    if not all_labels:
        raise SystemExit("No labels were produced; check species_qc.csv and the genomes.")

    df = pd.DataFrame(all_labels)
    df.to_csv(P('silver_labels.csv'), index=False)
    np.savez(P('silver_motifs.npz'), **all_pwms)
    pd.DataFrame(ortholog_rows).to_csv(P('silver_orthologs.csv'), index=False)

    pd.set_option('display.width', 240, 'display.max_columns', 25)
    print("Silver labels per species:")
    print(summarize(df).to_string(index=False))

    trainable = df[df['used_for_training']]
    n_species = trainable['species'].nunique()
    print(f"\n{len(trainable)} trainable labels across {n_species} species "
          f"(was 4 labels across 1 species).")

    if control_rows:
        ctrl = pd.DataFrame(control_rows)
        ctrl.to_csv(P('silver_label_control.csv'), index=False)
        print("\nControl — does this procedure recover the KNOWN C. elegans motifs?")
        print(ctrl.to_string(index=False))
        print(f"  mean skill vs measured motifs: {ctrl['skill_vs_known'].mean():+.3f}")
        print("  (this is the quality of the silver labels; near 0 or below means "
              "the labels are noise and the model built on them cannot be trusted.)")

    if n_species < 3:
        print("\nWarning: leave-one-species-out needs at least 3 species with "
              "trainable labels.")
    print(f"\nSaved {P('silver_labels.csv')}, {P('silver_motifs.npz')}")


if __name__ == '__main__':
    main()
