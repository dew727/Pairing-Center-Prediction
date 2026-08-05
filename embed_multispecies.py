"""Embed every species' ZIM/HIM-8 proteins with ESM-2.

Same frozen protein language model as before, applied to the whole cross-species
protein set instead of eight proteins. Two feature sets are written, because the
single-species work already showed they are not equally good:

  * **recognition-helix** — pooled over only the residues of the two atypical zinc
    fingers' recognition helices. Liu 2009 showed gene conversion has homogenised
    everything N-terminal to the fingers, so the rest of the protein carries no
    paralog-specific information. On the four-protein evaluation this feature was
    markedly sharper than whole-domain pooling (skill -0.077 vs -0.31) and it
    recovered the correct Cbr-HIM-8 assignment that whole-domain pooling missed.
    This is the default.
  * **domain** — pooled over the whole zinc-finger-to-C-terminus region, kept so the
    two can be compared on the larger training set.

The model reads the entire protein for context either way; the choice only affects
which residues are pooled into the final vector.

Run from the project root, after build_multispecies_labels.py:
    python embed_multispecies.py
Writes data/processed/12_multispecies/multispecies_embeddings_*.npz.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd

from src.paths import P
from src.data_io import read_fasta
from src.embeddings import (DEFAULT_MODEL, embed_positions, embed_sequences,
                            zinc_finger_domain)
from src.species import load_manifest
from src.zinc_fingers import recognition_positions
from find_orthologs import CE_PROTEINS, QUERY_LOCI, read_headers, representative_queries

ORTHOLOG_TABLE = P('silver_orthologs.csv')


def collect_proteins():
    """Gather every protein to embed: each species' orthologs, plus C. elegans.

    Returns records of (embedding key, species, paralog, protein id, sequence). The
    key is what the label table and the model use to look a protein up, so it is
    built the same way in both places: short species prefix + paralog.
    """
    manifest = {s.key: s for s in load_manifest()}
    records, seen = [], set()

    if not os.path.exists(ORTHOLOG_TABLE):
        raise SystemExit(f"No {ORTHOLOG_TABLE}. Run build_multispecies_labels.py first.")

    # The four C. elegans proteins: the queries, and the gold-standard test set.
    ce_path = next((s.proteins for s in manifest.values()
                    if 'elegans' in s.key.lower() and os.path.exists(s.proteins or '')),
                   CE_PROTEINS)
    if not os.path.exists(ce_path):
        raise SystemExit(
            f"C. elegans proteome not found at {ce_path}. It is needed twice over: as "
            f"the query set for finding orthologs, and as the gold test set. Download "
            f"it into data/raw/ and point the manifest at it.")
    ce_proteins = read_fasta(ce_path)
    ce_chosen = representative_queries(ce_proteins, read_headers(ce_path), QUERY_LOCI)
    for locus, (pid, seq, _) in ce_chosen.items():
        key = f'Cel_{locus}'
        records.append((key, 'c_elegans', locus.upper(), pid, seq))
        seen.add(key)

    table = pd.read_csv(ORTHOLOG_TABLE)

    proteomes = {}
    for _, r in table.iterrows():
        species = manifest.get(r['species'])
        if species is None:
            continue
        if r['species'] not in proteomes:
            proteomes[r['species']] = read_fasta(species.proteins)
        seq = proteomes[r['species']].get(r['protein_id'])
        if seq is None:
            continue
        key = f"{species.short}_{r['paralog'].lower()}"
        if key in seen:
            continue
        records.append((key, r['species'], r['paralog'], r['protein_id'], seq))
        seen.add(key)
    return records


def main():
    records = collect_proteins()
    keys = [r[0] for r in records]
    seqs = [r[4] for r in records]
    print(f"Embedding {len(records)} proteins with {DEFAULT_MODEL}\n")

    # Recognition-helix features: pool only the DNA-reading residues of the two fingers.
    positions, fallback = [], []
    for key, _, _, _, seq in records:
        helices = recognition_positions(seq, n_fingers=2)
        pos = [p for helix in helices for p in helix]
        if not pos:
            # No recognisable finger: pool the binding domain instead, and say so.
            span = zinc_finger_domain(seq) or (0, len(seq))
            pos = list(range(*span))
            fallback.append(key)
        positions.append(pos)
    if fallback:
        print(f"  no zinc finger found in {len(fallback)} protein(s); pooled the whole "
              f"binding region instead: {', '.join(fallback)}\n")

    recognition = embed_positions(seqs, positions, model_name=DEFAULT_MODEL)
    np.savez(P('multispecies_embeddings_recognition.npz'),
             **{k: recognition[i] for i, k in enumerate(keys)})

    # Whole-domain features, for comparison.
    regions = [zinc_finger_domain(s) or (0, len(s)) for s in seqs]
    domain = embed_sequences(seqs, model_name=DEFAULT_MODEL, regions=regions)
    np.savez(P('multispecies_embeddings_domain.npz'),
             **{k: domain[i] for i, k in enumerate(keys)})

    meta = {
        'model': DEFAULT_MODEL,
        'embedding_dim': int(recognition.shape[1]),
        'n_proteins': len(records),
        'feature_sets': {
            'recognition': 'mean-pooled over the two zinc fingers\' recognition helices',
            'domain': 'mean-pooled over the zinc-finger-to-C-terminus region',
        },
        'recognition_fallback': fallback,
        'created': datetime.date.today().isoformat(),
        'proteins': {k: {'species': sp, 'paralog': pa, 'protein_id': pid,
                         'length': len(seq)}
                     for (k, sp, pa, pid, seq) in records},
    }
    with open(P('multispecies_embeddings_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    per_species = pd.Series([r[1] for r in records]).value_counts()
    print("Proteins embedded per species:")
    print(per_species.to_string())
    print(f"\nSaved {P('multispecies_embeddings_recognition.npz')} "
          f"({len(keys)} x {recognition.shape[1]}) and the domain variant.")


if __name__ == '__main__':
    main()
