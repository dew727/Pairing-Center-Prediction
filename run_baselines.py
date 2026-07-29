"""Simple baseline predictors, scored with our one shared scoring method.

A baseline is a deliberately simple method that a real model has to beat to prove
it is worth anything.

Baseline 1 (in this file) is the "conserved-consensus" predictor: it always
predicts the part of the motif that is the same for every protein (the TTGG) and
just guesses randomly for the rest, completely ignoring the actual protein. Since
it puts no real effort into the positions we grade, it scores zero, which sets the
floor. Any genuine model must do better than this.

Baseline 2 (nearest-neighbour, in this file) predicts a held-out protein's motif
by copying the motif of the most similar OTHER C. elegans protein, judged by their
ESM-2 embeddings. It is the bar a learned model must clear to justify itself.

Baseline 3 (a small learned model) lives in train_head.py and is scored the same
way, so all three appear together in baseline_scores.csv.

Run this from the main project folder with: python run_baselines.py
It writes data/processed/baseline_scores.csv and, for the nearest-neighbour
method, data/processed/cb_predicted_motifs_nn.csv (the C. briggsae predictions).
"""
import numpy as np
from src.paths import P
import pandas as pd

from src.motifs import KNOWN_MOTIFS, BASES, motif_to_pwm, anchored_transfer
from src.metrics import evaluate, PARALOG_SPACER

DOMAIN_EMBEDDINGS = P('embeddings_domain.npz')


def conserved_consensus_pwm(motif_len, pseudocount=0.01):
    """Build the conserved-consensus prediction: confident TTGG at the start, and
    an even guess (each letter equally likely) at every other position.

    It only knows the TTGG that all these motifs share; it has no information
    about the letters that vary from protein to protein, so it just guesses there.
    """
    pwm = np.full((4, motif_len), 0.25)
    for j, base in enumerate('TTGG'):
        col = np.full(4, pseudocount)
        col[BASES.index(base)] += 1.0
        pwm[:, j] = col / col.sum()
    return pwm


def load_domain_embeddings(path=DOMAIN_EMBEDDINGS):
    """Load the per-protein binding-domain embeddings saved earlier.

    Keys look like 'Cel_zim-1' / 'Cbr_him-8'; each value is one vector of numbers
    summarizing that protein's DNA-binding region.
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


def cosine(a, b):
    """Similarity of two vectors: 1 = same direction, 0 = unrelated."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _cel_key(paralog):
    """Map a metric paralog name ('ZIM-1') to its embedding key ('Cel_zim-1')."""
    return f'Cel_{paralog.lower()}'


def nearest_neighbor_lopo(embeddings, center=True):
    """Baseline 2, scored honestly by leave-one-paralog-out cross-validation.

    For each C. elegans protein in turn, we hide it, then ask: of the three
    remaining proteins, which has the most similar binding-domain embedding? We
    predict the hidden protein's motif to be that most-similar protein's motif
    (carried into the hidden protein's frame by the shared TTGG/TG anchors), and
    grade it on the variable positions only.

    If center=True we first subtract the average of the THREE training proteins
    from every embedding. ESM-2 vectors share a large common component that
    otherwise dominates the similarity; removing it exposes the differences that
    actually distinguish the proteins. Crucially the average is taken over the
    training proteins only — never the held-out one — so no information about the
    protein being predicted leaks into its own prediction.
    """
    paralogs = list(KNOWN_MOTIFS)
    rows = []
    for held in paralogs:
        train = [p for p in paralogs if p != held]
        mean = (np.mean([embeddings[_cel_key(p)] for p in train], axis=0)
                if center else 0.0)
        vec = {p: embeddings[_cel_key(p)] - mean for p in paralogs}

        ranked = sorted(((cosine(vec[held], vec[p]), p) for p in train),
                        reverse=True)
        nn_cos, nn = ranked[0]

        pred = anchored_transfer(motif_to_pwm(KNOWN_MOTIFS[nn]),
                                 PARALOG_SPACER[nn],
                                 len(KNOWN_MOTIFS[held]), PARALOG_SPACER[held])
        res = evaluate(pred, motif_to_pwm(KNOWN_MOTIFS[held]), held)
        res['baseline'] = 'nearest-neighbor' + ('' if center else ' (raw)')
        res['neighbor'] = nn
        res['neighbor_cos'] = round(nn_cos, 3)
        rows.append(res)
    return rows


def predict_briggsae(embeddings):
    """Use the nearest-neighbour rule to predict each C. briggsae motif.

    This is the actual deliverable (no ground truth exists to score it): for each
    C. briggsae ortholog, find its most similar C. elegans protein and predict it
    binds that protein's motif. The C. elegans average is used for centering here
    (the C. briggsae proteins are separate from it, so nothing leaks).
    """
    paralogs = list(KNOWN_MOTIFS)
    mean = np.mean([embeddings[_cel_key(p)] for p in paralogs], axis=0)
    rows = []
    for cb_paralog in paralogs:
        cb_key = f'Cbr_{cb_paralog.lower()}'
        if cb_key not in embeddings:
            continue
        cb_vec = embeddings[cb_key] - mean
        ranked = sorted(((cosine(cb_vec, embeddings[_cel_key(p)] - mean), p)
                         for p in paralogs), reverse=True)
        nn_cos, nn = ranked[0]
        rows.append({
            'cb_ortholog_of': cb_paralog,
            'predicted_as': nn,
            'predicted_motif': KNOWN_MOTIFS[nn],
            'neighbor_cos': round(nn_cos, 3),
        })
    return rows


def main():
    rows = []
    print("Baseline 1 — conserved-consensus predictor (the floor)")
    print("=" * 64)
    for paralog, motif in KNOWN_MOTIFS.items():
        true_pwm = motif_to_pwm(motif)
        pred_pwm = conserved_consensus_pwm(len(motif))
        res = evaluate(pred_pwm, true_pwm, paralog)
        res['baseline'] = 'conserved-consensus'
        rows.append(res)
        print(f"  {paralog:6s} motif={motif:13s} spacer={PARALOG_SPACER[paralog]}bp "
              f"variable_pos={res['n_variable_positions']}  "
              f"distance={res['distance']:.3f}  skill={res['skill']:+.3f}")

    mean_skill = np.mean([r['skill'] for r in rows])
    mean_dist = np.mean([r['distance'] for r in rows])
    print(f"\n  MEAN over paralogs: distance={mean_dist:.3f}  skill={mean_skill:+.3f}")
    print("  (skill = 0.000 is expected — this baseline defines the metric's floor.)")

    # Baseline 2 — nearest-neighbour in embedding space (leave-one-paralog-out).
    embeddings = load_domain_embeddings()
    for center, tag in [(False, 'raw cosine'), (True, 'mean-centered')]:
        print(f"\nBaseline 2 — nearest-neighbour, {tag} (leave-one-paralog-out)")
        print("=" * 64)
        nn_rows = nearest_neighbor_lopo(embeddings, center=center)
        rows.extend(nn_rows)
        for r in nn_rows:
            print(f"  {r['paralog']:6s} nearest={r['neighbor']:6s} "
                  f"cos={r['neighbor_cos']:+.3f}  "
                  f"distance={r['distance']:.3f}  skill={r['skill']:+.3f}")
        print(f"\n  MEAN over paralogs: "
              f"distance={np.mean([r['distance'] for r in nn_rows]):.3f}  "
              f"skill={np.mean([r['skill'] for r in nn_rows]):+.3f}")

    # Sanity check on the scoring: a perfect prediction should score exactly 1.
    print("\nMetric self-check — perfect prediction (pred = true):")
    for paralog, motif in KNOWN_MOTIFS.items():
        true_pwm = motif_to_pwm(motif)
        res = evaluate(true_pwm, true_pwm, paralog)
        print(f"  {paralog:6s} distance={res['distance']:.3f}  skill={res['skill']:+.3f}")
    print("  (skill should be +1.000 everywhere — confirms the metric is wired correctly.)")

    cols = ['baseline', 'paralog', 'neighbor', 'neighbor_cos',
            'n_variable_positions', 'distance', 'skill', 'metric']
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(P('baseline_scores.csv'), index=False)
    print("\nSaved data/processed/baseline_scores.csv")

    # The actual C. briggsae predictions (nearest-neighbour rule; no ground truth).
    cb_rows = predict_briggsae(embeddings)
    cb_df = pd.DataFrame(cb_rows)
    cb_df.to_csv(P('cb_predicted_motifs_nn.csv'), index=False)
    print("\nC. briggsae nearest-neighbour predictions:")
    for r in cb_rows:
        print(f"  Cbr-{r['cb_ortholog_of']:6s} -> predicted as {r['predicted_as']:6s} "
              f"({r['predicted_motif']}), cos={r['neighbor_cos']:+.3f}")
    print("Saved data/processed/cb_predicted_motifs_nn.csv")


if __name__ == '__main__':
    main()
