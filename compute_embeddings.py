"""Turn each protein into a summary vector of numbers (its ESM-2 embedding).

We run the model once on the four C. elegans proteins (which the model will learn
from) and their four C. briggsae counterparts (which we will predict), then save
one vector per protein. We also save a small text file recording which model and
which proteins produced the vectors, so the results are easy to trace later and
we never have to re-run the model unnecessarily.

Run this from the main project folder with: python compute_embeddings.py
It writes data/processed/embeddings.npz and embeddings_meta.json.
"""
import datetime
from src.paths import P
import json

import numpy as np
import pandas as pd

from src.data_io import read_fasta
from src.embeddings import embed_sequences, embedding_dim, DEFAULT_MODEL
from find_orthologs import (
    CE_PROTEINS, CB_PROTEINS, QUERY_LOCI, read_headers, representative_queries,
)

ORTHOLOG_TABLE = P('ortholog_table.csv')
POOLING = 'mean'


def collect_proteins():
    """Gather the eight proteins to embed: the four C. elegans proteins and their
    four C. briggsae counterparts (from the ortholog table we built earlier).

    Returns, for each one, a short label, its species, which of the four it is,
    its ID, and its sequence.
    """
    ce_proteins = read_fasta(CE_PROTEINS)
    cb_proteins = read_fasta(CB_PROTEINS)
    ce_headers = read_headers(CE_PROTEINS)
    queries = representative_queries(ce_proteins, ce_headers, QUERY_LOCI)

    records = []
    for paralog in QUERY_LOCI:
        pid, seq, _ = queries[paralog]
        records.append((f'Cel_{paralog}', 'C. elegans', paralog, pid, seq))

    table = pd.read_csv(ORTHOLOG_TABLE)
    for _, r in table.iterrows():
        pid = r['cb_protein']
        records.append((f"Cbr_{r['ce_paralog']}", 'C. briggsae',
                        r['ce_paralog'], pid, cb_proteins[pid]))
    return records


def main():
    records = collect_proteins()
    labels = [r[0] for r in records]
    seqs = [r[4] for r in records]

    print(f"Embedding {len(records)} proteins with {DEFAULT_MODEL} "
          f"(pooling={POOLING})...")
    for label, species, paralog, pid, seq in records:
        print(f"  {label:10s} {species:12s} {paralog:6s} {pid:12s} {len(seq)} aa")

    vectors = embed_sequences(seqs, model_name=DEFAULT_MODEL, pooling=POOLING)
    dim = vectors.shape[1]

    np.savez(P('embeddings.npz'),
             **{label: vectors[i] for i, label in enumerate(labels)})

    meta = {
        'model': DEFAULT_MODEL,
        'pooling': POOLING,
        'embedding_dim': int(dim),
        'note': 'frozen ESM-2 feature extractor; swap model to esm2_t33_650M_UR50D later',
        'created': datetime.date.today().isoformat(),
        'proteins': {
            label: {'species': sp, 'paralog': pa, 'protein_id': pid, 'length': len(seq)}
            for (label, sp, pa, pid, seq) in records
        },
    }
    with open(P('embeddings_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved data/processed/embeddings.npz "
          f"({len(labels)} vectors x {dim} dims) and embeddings_meta.json")


if __name__ == '__main__':
    main()
