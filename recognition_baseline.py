"""Biology-informed features: embed only the zinc-finger recognition helices.

Motivation (Liu 2009, gene conversion): everything N-terminal to the two zinc
fingers has been homogenised between paralogs and carries no paralog-specific
signal, while DNA specificity is set by the residues in the fingers' recognition
helices. So instead of averaging ESM-2 over the whole binding domain (which blurs
those few residues), we read the embedding out at ONLY the recognition-helix
positions of each finger and concatenate the two fingers. Everything else — the
metric, the leave-one-paralog-out protocol, the leakage-safe centering — is
identical to the other baselines, so the scores are directly comparable.

Run from the project root:  python recognition_baseline.py
Writes data/processed/embeddings_recognition.npz and prints scores vs. the floor.
"""
import datetime
from src.paths import P
import json

import numpy as np

from src.embeddings import embed_positions, embed_sequences, DEFAULT_MODEL
from src.zinc_fingers import (recognition_positions, recognition_residues,
                             ctd_region, find_fingers)
from compute_embeddings import collect_proteins
from run_baselines import nearest_neighbor_lopo, predict_briggsae
from train_head import lopo_head

EMB_FILE = P('embeddings_recognition.npz')


def compute_embeddings():
    """Embed each protein's two recognition helices and concatenate them."""
    records = collect_proteins()
    labels = [r[0] for r in records]
    seqs = [r[4] for r in records]
    per_finger = [recognition_positions(s, n_fingers=2) for s in seqs]

    print("Recognition helices (ZF1 | ZF2):")
    for (label, *_ , seq), pf in zip(records, per_finger):
        r = recognition_residues(seq)
        print(f"  {label:10s} {r[0]:20s} | {r[1] if len(r) > 1 else '-'}")

    # Mean-pool the embedding over each finger's recognition positions, then
    # concatenate finger 1 and finger 2 so the two helices stay distinct.
    print(f"\nEmbedding recognition helices with {DEFAULT_MODEL}...")
    zf1 = embed_positions(seqs, [pf[0] for pf in per_finger])
    zf2 = embed_positions(seqs, [pf[1] for pf in per_finger])
    feats = np.hstack([zf1, zf2])
    emb = {label: feats[i] for i, label in enumerate(labels)}

    np.savez(EMB_FILE, **emb)
    json.dump({'model': DEFAULT_MODEL,
               'features': 'mean over ZF1 recognition helix ++ mean over ZF2 recognition helix',
               'embedding_dim': int(feats.shape[1]),
               'created': datetime.date.today().isoformat(),
               'source': 'Liu 2009 gene-conversion rationale: specificity is in the ZF recognition helices'},
              open(P('embeddings_recognition_meta.json'), 'w'), indent=2)
    print(f"Saved {EMB_FILE} ({feats.shape[0]} x {feats.shape[1]})")
    return emb


def region_features():
    """Embed three biologically-motivated regions of each protein for comparison:
    the zinc-finger recognition helices (gene-conversion paper), the CTD (Li 2024
    says specificity lives here), and the whole ZF1-2-CTD domain. Returns a dict of
    {region_name: {label: vector}}.
    """
    records = collect_proteins()
    labels = [r[0] for r in records]
    seqs = [r[4] for r in records]

    rec_pos = [recognition_positions(s, 2) for s in seqs]
    ctd = [ctd_region(s, 2) for s in seqs]
    dom = [(find_fingers(s)[0][0], len(s)) for s in seqs]  # first finger start -> C-terminus

    out = {}
    zf1 = embed_positions(seqs, [p[0] for p in rec_pos])
    zf2 = embed_positions(seqs, [p[1] for p in rec_pos])
    out['recognition-helices'] = {l: np.hstack([zf1[i], zf2[i]]) for i, l in enumerate(labels)}
    ctd_vecs = embed_sequences(seqs, regions=ctd)
    out['CTD-only'] = {l: ctd_vecs[i] for i, l in enumerate(labels)}
    dom_vecs = embed_sequences(seqs, regions=dom)
    out['ZF1-2-CTD (whole domain)'] = {l: dom_vecs[i] for i, l in enumerate(labels)}
    return out


def main():
    emb = compute_embeddings()

    # Paper-driven test: WHERE does the paralog-discriminating signal live?
    print("\n" + "=" * 64)
    print("Feature-localization test (Li 2024: specificity is set by the CTD)")
    print("nearest-neighbour mean skill (LOPO); floor = 0.000")
    print("=" * 64)
    for name, region_emb in region_features().items():
        nn = nearest_neighbor_lopo(region_emb, center=True)
        print(f"  {name:28s} mean skill={np.mean([r['skill'] for r in nn]):+.3f}")

    print("\n" + "=" * 64)
    print("Recognition-helix features — nearest-neighbour (leave-one-paralog-out)")
    print("=" * 64)
    nn = nearest_neighbor_lopo(emb, center=True)
    for r in nn:
        print(f"  {r['paralog']:6s} nearest={r['neighbor']:6s} "
              f"cos={r['neighbor_cos']:+.3f}  skill={r['skill']:+.3f}")
    print(f"  MEAN skill={np.mean([r['skill'] for r in nn]):+.3f}  (floor=0.000)")

    print("\n" + "=" * 64)
    print("Recognition-helix features — learned head (leave-one-paralog-out)")
    print("=" * 64)
    for l2 in [1, 3, 10]:
        rows = lopo_head(emb, projection=None, l2=l2)
        print(f"  L2={l2:>3}  MEAN skill={np.mean([r['skill'] for r in rows]):+.3f}")

    print("\nC. briggsae predictions (recognition-helix nearest neighbour):")
    for r in predict_briggsae(emb):
        print(f"  Cbr-{r['cb_ortholog_of']:6s} -> {r['predicted_as']:6s} "
              f"({r['predicted_motif']})  cos={r['neighbor_cos']:+.3f}")


if __name__ == '__main__':
    main()
