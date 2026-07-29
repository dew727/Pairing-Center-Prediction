"""Check whether focusing on the DNA-binding region gives better embeddings.

The question is whether the protein embeddings can tell the four proteins apart.
Summarizing the whole protein made them look nearly identical. Here we compare two
ways of summarizing each protein: using the whole thing, versus using only its
DNA-binding region. We test both with a simple check: for each C. elegans protein,
is its most similar C. briggsae protein its known counterpart? Whichever way gets
more of those right is the better summary.

Run this from the main project folder. It writes data/processed/embeddings_domain.npz.
"""
import datetime
from src.paths import P
import json

import numpy as np

from src.embeddings import embed_sequences, zinc_finger_domain, DEFAULT_MODEL
from compute_embeddings import collect_proteins


def cosine(a, b):
    """Similarity between two vectors: 1 means identical direction, 0 unrelated."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def report(name, labels, vectors):
    """For each C. elegans protein, check whether its most similar C. briggsae
    protein is its known counterpart, and print how well the two are separated."""
    E = {l: vectors[i] for i, l in enumerate(labels)}
    cel = [l for l in labels if l.startswith('Cel_')]
    cbr = [l for l in labels if l.startswith('Cbr_')]
    correct = 0
    diag, offdiag = [], []
    print(f"\n=== {name} ===")
    for c in cel:
        ortholog = c.replace('Cel_', 'Cbr_')
        sims = sorted(((cosine(E[c], E[b]), b) for b in cbr), reverse=True)
        nn = sims[0][1]
        hit = nn == ortholog
        correct += hit
        for s, b in sims:
            (diag if b == ortholog else offdiag).append(s)
        print(f"  {c:10s} -> {nn:10s} cos={sims[0][0]:.3f}"
              f"   ortholog cos={dict((b, s) for s, b in sims)[ortholog]:.3f}"
              f"   {'OK' if hit else 'miss'}")
    print(f"  nearest-neighbour orthologs correct: {correct}/{len(cel)}")
    print(f"  mean cosine  ortholog={np.mean(diag):.3f}  non-ortholog={np.mean(offdiag):.3f}"
          f"  separation={np.mean(diag) - np.mean(offdiag):+.3f}")
    return correct


def main():
    records = collect_proteins()
    labels = [r[0] for r in records]
    seqs = [r[4] for r in records]

    regions = [zinc_finger_domain(s) for s in seqs]
    print("Binding-domain (ZF1-ZF2-CTD) regions found:")
    for (label, sp, pa, pid, seq), reg in zip(records, regions):
        frac = (reg[1] - reg[0]) / len(seq)
        print(f"  {label:10s} {pid:12s} full={len(seq):4d}  domain={reg[0]}-{reg[1]} "
              f"({reg[1]-reg[0]} aa, {frac:.0%} of protein)")

    print(f"\nEmbedding with {DEFAULT_MODEL}...")
    full_vecs = embed_sequences(seqs, model_name=DEFAULT_MODEL)
    dom_vecs = embed_sequences(seqs, model_name=DEFAULT_MODEL, regions=regions)

    n_full = report("FULL-LENGTH (mean over whole protein)", labels, full_vecs)
    n_dom = report("DOMAIN-ONLY (mean over ZF1-ZF2-CTD)", labels, dom_vecs)

    # Save the DNA-binding-region embeddings for use in the later baselines.
    np.savez(P('embeddings_domain.npz'),
             **{l: dom_vecs[i] for i, l in enumerate(labels)})
    meta = {
        'model': DEFAULT_MODEL, 'pooling': 'mean over ZF1-ZF2-CTD domain',
        'embedding_dim': int(dom_vecs.shape[1]),
        'domain_finder': 'tandem C2H2 regex; region = [ZF1_start-5 : C-terminus]',
        'created': datetime.date.today().isoformat(),
        'regions': {labels[i]: [int(regions[i][0]), int(regions[i][1])]
                    for i in range(len(labels))},
    }
    with open(P('embeddings_domain_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print(f"VERDICT  full-length {n_full}/4  vs  domain-only {n_dom}/4 orthologs recovered")
    print("Saved data/processed/embeddings_domain.npz + _meta.json")


if __name__ == '__main__':
    main()
