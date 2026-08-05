"""Train and honestly evaluate the protein -> motif model on the multi-species set.

The previous version of this model was trained on four proteins and could only be
evaluated by leave-one-paralog-out, where nothing beat the uniform floor. With
labels from many genomes there is enough data to ask better questions, and this
script asks three of them:

  1. **Leave-one-species-out.** Hold out every species in turn. Can the model
     predict the motifs of a genome it has never seen?
  2. **Gold test.** Train on discovered labels only, test on the four *C. elegans*
     motifs that were measured in a lab. No C. elegans label is ever trained on, and
     the test labels did not come from this pipeline, so this number cannot be
     inflated by the label-generating procedure.
  3. **Does the protein matter at all?** Reported at every step by the
     group-consensus baseline, which ignores the embedding and predicts a protein's
     motif from its paralog group alone. Because the labels were joined to proteins
     by synteny, a model can score well simply by recognising the group. Beating the
     uniform floor is therefore *not* evidence of anything. Beating group-consensus
     is.

Run from the project root, after embed_multispecies.py:
    python train_multispecies.py [--domain] [--l2 1.0]
Writes data/processed/12_multispecies/multispecies_*.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

from src.paths import P
from src.motifs import KNOWN_MOTIFS
from src.metrics import PARALOG_SPACER
from src.pcmodel import (Example, build_examples, gold_test, leave_one_species_out,
                         summarize)

LABELS = P('silver_labels.csv')
DEFAULT_L2 = 1.0
L2_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30]


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def label_key(row):
    """A label's protein, named the way the embedding file names it."""
    return f"{row['short']}_{row['paralog'].lower()}"


def load_examples(embeddings, pwms):
    """Training examples from the silver labels, and the gold C. elegans test set."""
    if not os.path.exists(LABELS):
        raise SystemExit(f"No {LABELS}. Run build_multispecies_labels.py first.")
    df = pd.read_csv(LABELS)
    train_rows = df[df['used_for_training'] & (df['label_type'] == 'silver')]
    if train_rows.empty:
        raise SystemExit("No trainable silver labels. Check silver_labels.csv — every "
                         "discovery may have come out LOW-confidence.")

    examples, missing = build_examples(train_rows.to_dict('records'), embeddings,
                                       label_key, pwms=pwms)
    if missing:
        print(f"  {len(missing)} label(s) had no embedding and were skipped: "
              f"{sorted(set(missing))}")

    # The gold set: the four measured C. elegans motifs.
    gold = []
    for paralog, motif in KNOWN_MOTIFS.items():
        key = f'Cel_{paralog.lower()}'
        if key not in embeddings:
            print(f"  warning: no embedding for {key}; gold test will be incomplete")
            continue
        gold.append(Example(key=key, species='c_elegans', paralog=paralog,
                            vector=embeddings[key], motif=motif,
                            spacer=PARALOG_SPACER[paralog], weight=1.0,
                            label_type='gold'))
    return examples, gold


def _report(title, rows, note=''):
    """Print the mean skill of every method, best first, with the bar to beat named."""
    table = summarize(rows)
    print(f"\n{title}")
    print("=" * 74)
    if table.empty:
        print("  (no results)")
        return table
    for _, r in table.iterrows():
        cols = f"  {r['method']:20s} n={int(r['n']):3d}  skill={r['skill']:+.3f}"
        if 'native_skill' in table.columns and not np.isnan(r['native_skill']):
            cols += f"  (native frame {r['native_skill']:+.3f})"
        print(cols)
    bar = table.loc[table['method'] == 'group-consensus', 'skill']
    head = table.loc[table['method'] == 'learned-head', 'skill']
    if len(bar) and len(head):
        delta = float(head.iloc[0]) - float(bar.iloc[0])
        verdict = ("the head beats it — the protein embedding is contributing"
                   if delta > 0 else
                   "the head does NOT beat it — the embedding adds nothing here")
        print(f"\n  learned-head minus group-consensus: {delta:+.3f}  ->  {verdict}")
    if note:
        print(f"  {note}")
    return table


def main():
    l2 = _arg('--l2', DEFAULT_L2)
    feature = 'domain' if '--domain' in sys.argv else 'recognition'
    emb_path = P(f'multispecies_embeddings_{feature}.npz')
    if not os.path.exists(emb_path):
        raise SystemExit(f"No {emb_path}. Run embed_multispecies.py first.")

    embeddings = {k: v for k, v in np.load(emb_path).items()}
    pwm_path = P('silver_motifs.npz')
    pwms = {k: v for k, v in np.load(pwm_path).items()} if os.path.exists(pwm_path) else None

    print(f"Features: {feature} ESM-2 embeddings ({len(embeddings)} proteins), L2={l2}")
    examples, gold = load_examples(embeddings, pwms)
    species = sorted({e.species for e in examples})
    print(f"Training set: {len(examples)} labels across {len(species)} species "
          f"({', '.join(species)})")
    print(f"Gold test set: {len(gold)} measured C. elegans motifs")

    all_rows = []

    # 1. Leave-one-species-out.
    if len(species) >= 3:
        loso = leave_one_species_out(examples, l2=l2)
        for r in loso:
            r['evaluation'] = 'leave-one-species-out'
        all_rows += loso
        _report("Leave-one-species-out (predicting a genome the model never saw)", loso)

        per_species = (pd.DataFrame(loso)
                       .pivot_table(index='held_out_species', columns='method',
                                    values='skill', aggfunc='mean'))
        print("\n  Per held-out species (skill):")
        print(per_species.round(3).to_string())
    else:
        print(f"\nSkipping leave-one-species-out: needs 3+ species with trainable "
              f"labels, have {len(species)}.")

    # 2. Gold test — the number that actually settles it.
    if gold:
        gold_rows = gold_test(examples, gold, l2=l2)
        for r in gold_rows:
            r['evaluation'] = 'gold-test'
        all_rows += gold_rows
        _report("Gold test — trained on discovered labels, tested on measured motifs",
                gold_rows,
                note="native-frame skill is directly comparable to the previous "
                     "single-species result (floor 0.000, nearest-neighbour -0.31, "
                     "learned head -0.12).")

        print("\n  Per paralog (native-frame skill):")
        gd = pd.DataFrame(gold_rows)
        print(gd.pivot_table(index='paralog', columns='method',
                             values='native_skill', aggfunc='mean')
                .round(3).to_string())

    if all_rows:
        pd.DataFrame(all_rows).to_csv(P('multispecies_scores.csv'), index=False)
        summarize(pd.DataFrame(all_rows).to_dict('records')).to_csv(
            P('multispecies_summary.csv'), index=False)

    # 3. Regularisation sweep, so no conclusion rests on one L2 setting.
    if gold and len(species) >= 3:
        print("\nRegularisation sweep (mean skill; the head must beat group-consensus)")
        print("=" * 74)
        print(f"  {'L2':>6} | {'LOSO head':>10} {'LOSO group':>11} | "
              f"{'gold head':>10} {'gold group':>11}")
        sweep = []
        for g in L2_GRID:
            lo = summarize(leave_one_species_out(examples, l2=g)).set_index('method')
            go = summarize(gold_test(examples, gold, l2=g)).set_index('method')
            row = {
                'l2': g,
                'loso_head': round(float(lo.loc['learned-head', 'skill']), 3),
                'loso_group': round(float(lo.loc['group-consensus', 'skill']), 3),
                'gold_head': round(float(go.loc['learned-head', 'skill']), 3),
                'gold_group': round(float(go.loc['group-consensus', 'skill']), 3),
            }
            sweep.append(row)
            print(f"  {g:>6} | {row['loso_head']:>+10.3f} {row['loso_group']:>+11.3f} | "
                  f"{row['gold_head']:>+10.3f} {row['gold_group']:>+11.3f}")
        pd.DataFrame(sweep).to_csv(P('multispecies_l2_sweep.csv'), index=False)

    print(f"\nSaved {P('multispecies_scores.csv')} and {P('multispecies_summary.csv')}")


if __name__ == '__main__':
    main()
