"""The protein -> motif model, retrained on many species instead of four proteins.

This is the same small head as `train_head.py` (one 4-way softmax per canonical
motif slot), changed in three ways that the extra data makes possible and necessary:

  * **Weighted examples.** Silver labels are discovered motifs, not measured ones,
    so each carries a weight reflecting how confident the discovery was. A shaky
    label nudges the model; a strong one moves it.
  * **Soft targets.** A discovered motif comes with a PWM, not just a consensus
    string, so we train against the letter *distribution* actually observed in the
    tandem array. Positions the array is unsure about teach the model less, which is
    exactly right.
  * **Leave-one-species-out.** With four proteins the only possible split was
    leave-one-paralog-out. With many species we can hold out an entire species,
    which asks the question that actually matters: given a protein from a genome the
    model has never seen, can it predict that protein's motif?

The baselines are the important part of this file. A model trained on labels that
were assigned to proteins *by synteny* can score well for a hollow reason: it can
learn "which of the four paralog groups is this protein?" and emit that group's
usual motif, having learned nothing about the protein's own sequence. So the bar to
beat is not the uniform floor, it is the **group-consensus baseline**, which does
exactly that hollow thing on purpose. A head that cannot beat it has not learned
anything from the protein language model.
"""
import numpy as np

from src.metrics import UNIFORM_COLUMN, column_distance, variable_mask
from src.motifs import (BASES, motif_to_pwm, pwm_to_canonical, canonical_to_pwm,
                        CANON_VARIABLE)

_BASE_IDX = {b: i for i, b in enumerate(BASES)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
#
# The project's locked metric is: mean per-column distance over the *variable*
# positions (never the shared TTGG/TG anchors, which every method gets free),
# rescaled so 0 = a blind uniform guess and 1 = the exact motif.
#
# Across species the motifs differ in length, so we apply that same definition in
# the shared canonical frame (the 9 biologically-aligned slots) rather than in one
# motif's own columns. `score_native` keeps the original per-motif version, so the
# C. elegans test numbers stay directly comparable to the previous results.

def score_canonical(pred_canon, true_canon, mask, metric='euclidean'):
    """Score a prediction in the shared 9-slot canonical frame.

    Only the slots the protein actually uses (`mask`) are graded, so a protein with a
    short spacer is not penalised for slots it does not have.
    """
    slots = np.where(mask)[0]
    if len(slots) == 0:
        return {'distance': float('nan'), 'skill': float('nan'), 'n_slots': 0}
    pred_d = float(np.mean([column_distance(pred_canon[s], true_canon[s], metric)
                            for s in slots]))
    base_d = float(np.mean([column_distance(UNIFORM_COLUMN, true_canon[s], metric)
                            for s in slots]))
    return {'distance': pred_d,
            'skill': 1.0 - pred_d / base_d if base_d > 0 else 0.0,
            'n_slots': int(len(slots))}


def score_native(pred_pwm, true_pwm, spacer, metric='euclidean'):
    """Score in the motif's own frame — the original metric, for comparability."""
    mask = variable_mask(true_pwm.shape[1], spacer)
    cols = np.where(mask)[0]
    pred_d = float(np.mean([column_distance(pred_pwm[:, j], true_pwm[:, j], metric)
                            for j in cols]))
    base_d = float(np.mean([column_distance(UNIFORM_COLUMN, true_pwm[:, j], metric)
                            for j in cols]))
    return {'distance': pred_d,
            'skill': 1.0 - pred_d / base_d if base_d > 0 else 0.0,
            'n_variable_positions': int(len(cols))}


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

class Example:
    """One training pair: a protein's embedding and the motif it binds.

    The motif is held in the shared canonical frame — a (9, 4) table of letter
    probabilities plus a mask of which of the 9 slots this protein uses — so
    motifs of different lengths and spacers can train one model together.
    """

    def __init__(self, key, species, paralog, vector, motif, spacer,
                 pwm=None, weight=1.0, label_type='silver'):
        self.key = key
        self.species = species
        self.paralog = paralog
        self.vector = np.asarray(vector, float)
        self.motif = motif
        self.spacer = int(spacer)
        self.weight = float(weight)
        self.label_type = label_type
        # Soft targets where a discovered PWM is available, otherwise the consensus.
        source = pwm if pwm is not None else motif_to_pwm(motif)
        self.pwm = source
        self.canon, self.mask = pwm_to_canonical(source, self.spacer)


def build_examples(rows, embeddings, key_fn, pwms=None):
    """Assemble Examples from a label table plus the embedding file.

    Labels whose protein has no embedding are skipped and reported by the caller
    rather than silently dropped.
    """
    examples, missing = [], []
    for r in rows:
        key = key_fn(r)
        if key not in embeddings:
            missing.append(key)
            continue
        pwm = None
        if pwms is not None:
            pwm_key = f"{r['species']}:{r['chromosome']}"
            if pwm_key in pwms:
                pwm = pwms[pwm_key]
        examples.append(Example(
            key=key, species=r['species'], paralog=r['paralog'],
            vector=embeddings[key], motif=r['motif'], spacer=r['spacer'],
            pwm=pwm, weight=float(r.get('weight', 1.0)),
            label_type=r.get('label_type', 'silver')))
    return examples, missing


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def prepare_features(examples, train_idx):
    """Center and scale embeddings using the TRAINING examples only.

    ESM-2 vectors share a large common component that swamps cosine similarity and
    dominates a linear head; subtracting the training mean exposes the differences
    that actually distinguish proteins. Both the mean and the scale come from the
    training split alone, so nothing about a held-out protein informs its own
    prediction. The scale keeps the L2 strength meaning the same thing regardless of
    the embedding's dimensionality.
    """
    train_vecs = np.vstack([examples[i].vector for i in train_idx])
    mean = train_vecs.mean(axis=0)
    scale = float(np.mean(np.linalg.norm(train_vecs - mean, axis=1))) + 1e-8
    return np.vstack([(e.vector - mean) / scale for e in examples])


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# The learned head
# ---------------------------------------------------------------------------

def train_head(features, examples, train_idx, l2=1.0, epochs=800, lr=None):
    """Fit one weighted 4-way softmax per canonical slot.

    Full-batch gradient descent with L2 regularisation. Each example contributes in
    proportion to its label weight, and the target is a letter *distribution* rather
    than a single letter, so an uncertain discovered motif teaches proportionally
    less. As L2 grows the head falls back to predicting the weighted average letter
    composition of the training set, which is the sensible thing to do when the
    protein carries no usable signal.
    """
    if lr is None:
        lr = 1.0 / (1.0 + l2)
    dim = features.shape[1]
    W = np.zeros((CANON_VARIABLE, 4, dim))
    b = np.zeros((CANON_VARIABLE, 4))

    for s in range(CANON_VARIABLE):
        rows = [i for i in train_idx
                if examples[i].mask[s] and examples[i].weight > 0]
        if not rows:
            continue
        X = features[rows]
        T = np.vstack([examples[i].canon[s] for i in rows])
        w = np.array([examples[i].weight for i in rows])
        w_sum = w.sum()
        Ws, bs = np.zeros((4, dim)), np.zeros(4)
        for _ in range(epochs):
            Pr = softmax(X @ Ws.T + bs)
            resid = (Pr - T) * w[:, None]
            Ws -= lr * (resid.T @ X / w_sum + l2 * Ws)
            bs -= lr * (resid.sum(axis=0) / w_sum)      # bias unregularised
        W[s], b[s] = Ws, bs
    return W, b


def predict_canonical(W, b, x):
    """Predict the 9 canonical-slot letter distributions for one protein."""
    return softmax(np.einsum('skd,d->sk', W, x) + b)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def uniform_canonical():
    """The floor: no information at any variable slot. Scores 0 by construction."""
    return np.full((CANON_VARIABLE, 4), 0.25)


def consensus_canonical(examples, idx):
    """The weighted average motif of a set of examples, in the canonical frame.

    Used for two baselines: averaged over one paralog group it is the
    group-consensus predictor; averaged over everything it is the global consensus.
    """
    canon = np.full((CANON_VARIABLE, 4), 0.25)
    for s in range(CANON_VARIABLE):
        rows = [i for i in idx if examples[i].mask[s] and examples[i].weight > 0]
        if not rows:
            continue
        w = np.array([examples[i].weight for i in rows])
        stack = np.vstack([examples[i].canon[s] for i in rows])
        canon[s] = (stack * w[:, None]).sum(axis=0) / w.sum()
    return canon


def group_consensus_canonical(examples, train_idx, paralog):
    """Predict a protein's motif as its paralog group's average training motif.

    This is the baseline that matters. The silver labels were joined to proteins by
    synteny — chromosome X's motif was assigned to HIM-8 because HIM-8 binds the X —
    so a model can appear to succeed just by recognising which paralog group a
    protein belongs to. This predictor does precisely that and nothing else. Beating
    it is the minimum evidence that the protein language model contributed anything.
    """
    rows = [i for i in train_idx if examples[i].paralog == paralog]
    if not rows:
        rows = list(train_idx)              # unseen group: fall back to global
    return consensus_canonical(examples, rows)


def nearest_neighbour_canonical(features, examples, train_idx, held):
    """Copy the motif of the most similar training protein (cosine on embeddings)."""
    x = features[held]
    best, best_cos = None, -np.inf
    for i in train_idx:
        if examples[i].weight <= 0:
            continue
        v = features[i]
        denom = np.linalg.norm(x) * np.linalg.norm(v)
        cos = float(x @ v / denom) if denom else -np.inf
        if cos > best_cos:
            best, best_cos = i, cos
    if best is None:
        return uniform_canonical(), None, float('nan')
    return examples[best].canon.copy(), examples[best], best_cos


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(examples, train_idx, test_idx, l2=1.0, epochs=800,
                   include_native=True):
    """Train on one split and score every method on the held-out examples.

    Returns one row per held-out example per method, all graded by the same locked
    metric in the shared canonical frame (plus, optionally, in the motif's own frame
    so numbers stay comparable to the single-species results).
    """
    features = prepare_features(examples, train_idx)
    W, b = train_head(features, examples, train_idx, l2=l2, epochs=epochs)
    global_consensus = consensus_canonical(examples, train_idx)

    rows = []
    for i in test_idx:
        e = examples[i]
        preds = {
            'floor (uniform)': uniform_canonical(),
            'global-consensus': global_consensus,
            'group-consensus': group_consensus_canonical(examples, train_idx, e.paralog),
            'nearest-neighbour': nearest_neighbour_canonical(
                features, examples, train_idx, i)[0],
            'learned-head': predict_canonical(W, b, features[i]),
        }
        for method, canon in preds.items():
            res = score_canonical(canon, e.canon, e.mask)
            row = {'method': method, 'species': e.species, 'paralog': e.paralog,
                   'protein': e.key, 'motif': e.motif, 'spacer': e.spacer,
                   'label_type': e.label_type, 'weight': e.weight,
                   'canonical_distance': round(res['distance'], 4),
                   'skill': round(res['skill'], 4), 'n_slots': res['n_slots']}
            if include_native:
                pred_pwm = canonical_to_pwm(canon, e.spacer, len(e.motif),
                                            anchor_motif=e.motif)
                nat = score_native(pred_pwm, e.pwm, e.spacer)
                row['native_skill'] = round(nat['skill'], 4)
            rows.append(row)
    return rows


def leave_one_species_out(examples, l2=1.0, epochs=800, min_train_species=2):
    """Hold out each species in turn, train on the others, score the held-out one.

    This is the evaluation the extra genomes buy us. Leave-one-paralog-out could only
    ever ask "can three proteins predict a fourth"; this asks "can a set of species
    predict a species the model has never seen", which is the question a usable
    predictor has to answer.
    """
    species = sorted({e.species for e in examples})
    if len(species) < min_train_species + 1:
        raise ValueError(
            f"leave-one-species-out needs at least {min_train_species + 1} species "
            f"with usable labels; got {len(species)}: {species}")
    rows = []
    for held in species:
        test_idx = [i for i, e in enumerate(examples) if e.species == held]
        train_idx = [i for i, e in enumerate(examples)
                     if e.species != held and e.weight > 0]
        if not train_idx or not test_idx:
            continue
        for row in evaluate_split(examples, train_idx, test_idx, l2=l2, epochs=epochs):
            row['held_out_species'] = held
            rows.append(row)
    return rows


def gold_test(examples, gold_examples, l2=1.0, epochs=800):
    """Train only on discovered (silver) labels, then test on the measured ones.

    The cleanest test available: the four *C. elegans* motifs come from SELEX and
    crystal structures, so they were never produced by this pipeline, and no
    *C. elegans* label is in the training set. If the head beats the group-consensus
    baseline here, the protein language model contributed real information.
    """
    combined = list(examples) + list(gold_examples)
    train_idx = [i for i in range(len(examples)) if examples[i].weight > 0]
    test_idx = list(range(len(examples), len(combined)))
    return evaluate_split(combined, train_idx, test_idx, l2=l2, epochs=epochs)


def summarize(rows, by='method'):
    """Mean skill per method — the table the whole exercise is judged on."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = {'skill': 'mean', 'canonical_distance': 'mean', 'protein': 'size'}
    if 'native_skill' in df.columns:
        agg['native_skill'] = 'mean'
    out = df.groupby(by).agg(agg).rename(columns={'protein': 'n'}).reset_index()
    return out.sort_values('skill', ascending=False)
