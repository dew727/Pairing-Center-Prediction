"""Baseline 3 — a small learned model from protein embedding to DNA motif.

Where the nearest-neighbour baseline just copies another protein's motif, this
one actually learns a mapping: given a protein's binding-domain embedding, predict
the letter probabilities at each variable position of its motif. The head is
deliberately tiny (a separate 4-way softmax per canonical position), because we
have only four labelled proteins and anything larger would simply memorise them.

Two settings are reported, both scored by the same leave-one-paralog-out (LOPO)
cross-validation and the same locked metric as the other baselines:

  * fine-tune-only : train the head on the 3 training proteins alone.
  * pretrained     : first learn a "binding-relevant" projection of the ESM-2
                     embedding from hundreds of unrelated transcription factors
                     (JASPAR; see pretrain_jaspar.py), then train the same small
                     head in that projected space. This is the setting that is
                     supposed to make four examples go further.

Run from the project root:  python train_head.py
Writes data/processed/head_scores.csv and updates the combined comparison.
"""
import json
from src.paths import P
import os

import numpy as np
import pandas as pd

from src.motifs import (KNOWN_MOTIFS, BASES, motif_to_pwm, pwm_to_canonical,
                        canonical_to_pwm, CANON_VARIABLE)
from src.metrics import evaluate, PARALOG_SPACER
from run_baselines import load_domain_embeddings, _cel_key

PROJECTION_FILE = P('jaspar_projection.npz')
_BASE_IDX = {b: i for i, b in enumerate(BASES)}


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def build_examples(paralogs, feats):
    """Assemble training targets: for each protein, which canonical slots it uses
    and the correct DNA letter (0..3) in each. `feats[p]` is that protein's
    (already projected/centered) feature vector.
    """
    examples = []
    for p in paralogs:
        motif = KNOWN_MOTIFS[p]
        canon, mask = pwm_to_canonical(motif_to_pwm(motif), PARALOG_SPACER[p])
        slots = {int(s): int(canon[s].argmax()) for s in np.where(mask)[0]}
        examples.append((feats[p], slots))
    return examples


def train_head(examples, dim, l2=10.0, epochs=1500, lr=None):
    """Fit one 4-way softmax classifier per canonical slot, sharing the input.

    Full-batch gradient descent with strong L2 regularisation — with only three
    training proteins the regulariser is what keeps the head from memorising, and
    in the limit it falls back to predicting the average letter composition the
    training proteins show at each slot. The learning rate is scaled down as the
    regularisation grows so the weight-decay step stays stable at any L2.
    """
    if lr is None:
        lr = 1.0 / (1.0 + l2)
    W = np.zeros((CANON_VARIABLE, 4, dim))
    b = np.zeros((CANON_VARIABLE, 4))
    # Group training instances by canonical slot.
    per_slot = {s: [] for s in range(CANON_VARIABLE)}
    for x, slots in examples:
        for s, base in slots.items():
            per_slot[s].append((x, base))

    for s, inst in per_slot.items():
        if not inst:
            continue
        X = np.vstack([x for x, _ in inst])
        y = np.array([base for _, base in inst])
        Ws, bs = np.zeros((4, dim)), np.zeros(4)
        onehot = np.eye(4)[y]
        for _ in range(epochs):
            P = softmax(X @ Ws.T + bs)
            gW = (P - onehot).T @ X / len(y) + l2 * Ws
            gb = (P - onehot).mean(axis=0)          # bias unregularised
            Ws -= lr * gW
            bs -= lr * gb
        W[s], b[s] = Ws, bs
    return W, b


def predict_canonical(W, b, x):
    """Predict the 9 canonical-slot letter distributions for one protein."""
    return softmax(np.einsum('skd,d->sk', W, x) + b)


def lopo_head(embeddings, projection=None, l2=10.0):
    """Leave-one-paralog-out evaluation of the learned head.

    For each held-out protein: center on the training mean (leakage-safe), apply
    the optional pretrained projection, train the head on the other three, then
    predict the held-out protein's motif and score it.
    """
    paralogs = list(KNOWN_MOTIFS)
    rows = []
    for held in paralogs:
        train = [p for p in paralogs if p != held]
        mean = np.mean([embeddings[_cel_key(p)] for p in train], axis=0)

        def feat(p):
            v = embeddings[_cel_key(p)] - mean
            return projection.T @ v if projection is not None else v

        raw = {p: feat(p) for p in paralogs}
        # Scale features to typical unit norm using the TRAINING proteins only, so
        # the head's L2 strength means the same thing whether the features are the
        # raw 1280-d embedding or the low-dimensional pretrained projection.
        scale = np.mean([np.linalg.norm(raw[p]) for p in train]) + 1e-8
        feats = {p: raw[p] / scale for p in paralogs}
        dim = feats[held].shape[0]
        examples = build_examples(train, feats)
        W, b = train_head(examples, dim, l2=l2)

        canon = predict_canonical(W, b, feats[held])
        pred = canonical_to_pwm(canon, PARALOG_SPACER[held],
                                len(KNOWN_MOTIFS[held]), anchor_motif=KNOWN_MOTIFS[held])
        res = evaluate(pred, motif_to_pwm(KNOWN_MOTIFS[held]), held)
        res['paralog'] = held
        rows.append(res)
    return rows


DEFAULT_L2 = 10.0


def main():
    embeddings = load_domain_embeddings()
    proj = (np.load(PROJECTION_FILE)['projection']
            if os.path.exists(PROJECTION_FILE) else None)
    settings = [('learned-head (fine-tune only)', None)]
    if proj is not None:
        settings.append((f'learned-head (pretrained, dim {proj.shape[1]})', proj))
    else:
        print(f"(No {PROJECTION_FILE} yet — run pretrain_jaspar.py for the "
              f"pretrained setting.)\n")

    all_rows = []
    for name, projection in settings:
        print(f"Baseline 3 — {name} (leave-one-paralog-out, L2={DEFAULT_L2})")
        print("=" * 64)
        rows = lopo_head(embeddings, projection=projection, l2=DEFAULT_L2)
        for r in rows:
            r['baseline'] = name
            r['l2'] = DEFAULT_L2
            print(f"  {r['paralog']:6s} distance={r['distance']:.3f}  "
                  f"skill={r['skill']:+.3f}")
        print(f"\n  MEAN: distance={np.mean([r['distance'] for r in rows]):.3f}  "
              f"skill={np.mean([r['skill'] for r in rows]):+.3f}\n")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)[
        ['baseline', 'paralog', 'l2', 'n_variable_positions', 'distance', 'skill', 'metric']]
    df.to_csv(P('head_scores.csv'), index=False)
    print("Saved data/processed/head_scores.csv")

    # Transparency: how mean skill depends on regularisation strength, so the
    # "never beats the floor" conclusion can be seen not to hinge on one L2.
    print("\nRegularisation sweep (mean skill over paralogs; floor = 0.000):")
    print(f"  {'L2':>7} | {'fine-tune only':>14} | {'pretrained':>12}")
    sweep = []
    for l2 in [0.1, 0.3, 1, 3, 10, 30, 100, 300]:
        ft = np.mean([r['skill'] for r in lopo_head(embeddings, None, l2=l2)])
        row = {'l2': l2, 'finetune_only': round(ft, 3)}
        line = f"  {l2:>7} | {ft:>+14.3f} |"
        if proj is not None:
            pt = np.mean([r['skill'] for r in lopo_head(embeddings, proj, l2=l2)])
            row['pretrained'] = round(pt, 3)
            line += f" {pt:>+12.3f}"
        sweep.append(row)
        print(line)
    pd.DataFrame(sweep).to_csv(P('head_l2_sweep.csv'), index=False)
    print("Saved data/processed/head_l2_sweep.csv")


if __name__ == '__main__':
    main()
