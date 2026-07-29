"""The single scoring method used to grade every prediction.

We deliberately keep ONE scoring function that every baseline and model uses, so
that different approaches can be compared fairly. If each method were graded
differently, the comparison would be meaningless.

What it measures: how close a predicted motif is to the true motif. Crucially, it
only grades the parts of the motif that actually differ between the proteins.
Every one of these motifs starts with TTGG and has a TG shortly after; any method
gets those right for free, so grading them would make every method look good. The
real challenge is the "spacer" letters between TTGG and TG, and the "tail" after
TG, so those are the only positions we grade.

Layout of a motif:

    T T G G  [ spacer ]  T G  [ tail ]

The spacer length differs by protein (Li et al. 2024): 4 letters for HIM-8 and
ZIM-2, 2 for ZIM-1, 1 for ZIM-3. So the TG lands at position 4 + spacer.

Motifs are compared as PWMs (the probability tables described in motifs.py). The
predicted and true PWM must describe motifs of the same length, lined up the same
way; it's up to the caller to arrange that.
"""
import numpy as np

# How many letters sit between TTGG and TG for each protein.
PARALOG_SPACER = {'HIM-8': 4, 'ZIM-1': 2, 'ZIM-2': 4, 'ZIM-3': 1}

# A column with no information: each of the four letters equally likely.
UNIFORM_COLUMN = np.full(4, 0.25)


def anchor_columns(spacer):
    """List the positions of the shared TTGG and TG that we do NOT grade."""
    tg = 4 + spacer
    return {0, 1, 2, 3, tg, tg + 1}


def variable_mask(motif_len, spacer):
    """Mark which positions of the motif we actually grade (everything except
    the shared TTGG and TG). Returns a True/False flag for each position."""
    conserved = anchor_columns(spacer)
    return np.array([j not in conserved for j in range(motif_len)])


def column_distance(p, q, metric='euclidean'):
    """Measure how different the predicted and true letter-probabilities are at
    a single position (p and q each hold four numbers, one per DNA letter).

    'euclidean' is straight-line distance between the two sets of four numbers.
    'jsd' (Jensen-Shannon) is an alternative that compares them as probability
    distributions. Either way, 0 means identical and larger means more different.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    if metric == 'euclidean':
        return float(np.sqrt(np.sum((p - q) ** 2)))
    if metric == 'jsd':
        m = 0.5 * (p + q)
        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    raise ValueError(f"unknown metric: {metric}")


def pwm_distance(pred_pwm, true_pwm, mask, metric='euclidean'):
    """Average the per-position difference across the graded positions only.

    This is the main score: lower means the prediction is closer to the truth.
    """
    cols = np.where(mask)[0]
    return float(np.mean([
        column_distance(pred_pwm[:, j], true_pwm[:, j], metric) for j in cols]))


def skill_score(pred_pwm, true_pwm, mask, metric='euclidean'):
    """Rescale the distance into an easy-to-read score between 0 and 1.

    0 means the prediction is no better than a blind guess (equal odds for every
    letter, which is what the conserved-consensus baseline does). 1 means it
    matches the true motif exactly. Higher is better, and because it is rescaled,
    it is comparable across proteins even though their motifs differ in length.
    """
    pred_d = pwm_distance(pred_pwm, true_pwm, mask, metric)
    cols = np.where(mask)[0]
    base_d = float(np.mean([
        column_distance(UNIFORM_COLUMN, true_pwm[:, j], metric) for j in cols]))
    return 1.0 - pred_d / base_d if base_d > 0 else 0.0


def evaluate(pred_pwm, true_pwm, paralog, metric='euclidean'):
    """Grade one prediction for one protein and return the results together.

    Figures out which positions to grade from the protein's spacer length, then
    reports the distance, the 0-to-1 skill score, and how many positions were
    graded.
    """
    spacer = PARALOG_SPACER[paralog]
    mask = variable_mask(true_pwm.shape[1], spacer)
    return {
        'paralog': paralog,
        'n_variable_positions': int(mask.sum()),
        'distance': pwm_distance(pred_pwm, true_pwm, mask, metric),
        'skill': skill_score(pred_pwm, true_pwm, mask, metric),
        'metric': metric,
    }
