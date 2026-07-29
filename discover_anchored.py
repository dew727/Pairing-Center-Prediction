"""Protein-anchored, drift-AWARE reconciliation of the two discovery views.

The two pipelines answer different questions:
  * discover_motifs.py  -- discovery CONSTRAINED to the C. elegans TTGG framework.
  * discover_denovo.py  -- anchor-FREE discovery, selected by chromosome-specificity.

Neither is right alone: hardcoding TTGG is circular across species, but pure architecture is
under-determined (chromosome ends carry many specific tandem repeats). The tie-breaker is the
TARGET species' OWN proteins: if a paralog's zinc-finger recognition helices are conserved
relative to C. elegans, it still reads the same anchor (TTGG), so the anchored motif is the
credible pairing center; if they have diverged, the anchor may have drifted, so the anchor-free
candidate becomes a genuine divergence hypothesis. Crucially this prior is derived from the
target ortholog itself -- not imposed from C. elegans -- and only the ANCHOR is gated; the
spacer/tail are always free (the spacer is CTD-set, which is where drift actually lives).

This script scores recognition-helix conservation per paralog and produces one drift-aware
sheet pairing the anchored and anchor-free candidates with a per-chromosome recommendation.

Run:  python discover_anchored.py
Writes data/processed/anchored_predictions.csv.
"""
import numpy as np
from src.paths import P
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

from compute_embeddings import collect_proteins
from src.zinc_fingers import recognition_residues
from discover_motifs import MAIN_CHROMS, CE_CHROM_PARALOG

CONS_HIGH, CONS_LOW = 0.60, 0.40      # anchor-conserved vs anchor-diverged thresholds


def _aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score = -10, -0.5
    return a


def _identity(a, b, aligner):
    if not a or not b:
        return 0.0
    al = aligner.align(a, b)[0]
    return str(al).split('\n')[1].count('|') / max(len(a), len(b))


def recognition_conservation(cel_prefix='Cel_', tgt_prefix='Cbr_'):
    """Mean recognition-helix identity (both fingers) between each C. elegans paralog and its
    target-species ortholog. High => the ortholog still reads the same DNA anchor."""
    recs = {r[0]: r[4] for r in collect_proteins()}
    aligner = _aligner()
    out = {}
    for pa in ['zim-1', 'zim-2', 'zim-3', 'him-8']:
        ce = recognition_residues(recs.get(f'{cel_prefix}{pa}', ''))
        tg = recognition_residues(recs.get(f'{tgt_prefix}{pa}', ''))
        ids = [_identity(ce[i], tg[i], aligner)
               for i in range(min(len(ce), len(tg)))]
        out[pa.upper().replace('ZIM', 'ZIM').replace('HIM', 'HIM')] = \
            round(float(np.mean(ids)) if ids else 0.0, 2)
    # normalise keys to metric style (ZIM-1, HIM-8)
    return {k.upper(): v for k, v in
            {'ZIM-1': out.get('ZIM-1'), 'ZIM-2': out.get('ZIM-2'),
             'ZIM-3': out.get('ZIM-3'), 'HIM-8': out.get('HIM-8')}.items()}


def main():
    cons = recognition_conservation()
    anchored = pd.read_csv(P('discovered_motifs_qx1410.csv')).set_index('chromosome')
    free = pd.read_csv(P('denovo_motifs_qx1410.csv')).set_index('chromosome')

    rows = []
    for c in MAIN_CHROMS:
        pa = CE_CHROM_PARALOG[c]
        k = cons.get(pa)
        if k is None:
            continue
        anchor_status = ('CONSERVED' if k >= CONS_HIGH else
                         'DIVERGED' if k < CONS_LOW else 'INTERMEDIATE')
        rec = {'CONSERVED': 'TTGG anchor supported by conserved protein -> trust the anchored motif',
               'DIVERGED': 'protein diverged -> anchor may have drifted; anchor-free candidate is a hypothesis',
               'INTERMEDIATE': 'both candidates plausible; needs experimental disambiguation'}[anchor_status]
        a = anchored.loc[c] if c in anchored.index else None
        f = free.loc[c] if c in free.index else None
        rows.append({
            'chromosome': c, 'paralog': f'Cbr-{pa}',
            'recognition_conservation': k, 'anchor_status': anchor_status,
            'anchored_motif': a['discovered_motif'] if a is not None else None,
            'anchored_skill': a['ce_match_skill'] if a is not None else None,
            'anchored_confidence': a['confidence'] if a is not None else None,
            'anchorfree_motif': f['discovered_motif'] if f is not None else None,
            'anchorfree_specificity': f['chrom_specificity'] if f is not None else None,
            'anchorfree_has_TTGG': f['recovered_TTGG'] if f is not None else None,
            'recommendation': rec,
        })
    out = pd.DataFrame(rows).sort_values('recognition_conservation', ascending=False)
    out.to_csv(P('anchored_predictions.csv'), index=False)

    pd.set_option('display.width', 300, 'display.max_columns', 30)
    print("Protein-anchored, drift-aware C. briggsae PC-motif predictions:\n")
    show = ['chromosome', 'paralog', 'recognition_conservation', 'anchor_status',
            'anchored_motif', 'anchored_skill', 'anchorfree_motif',
            'anchorfree_specificity', 'anchorfree_has_TTGG']
    print(out[show].to_string(index=False))
    print("\nPer-chromosome recommendation:")
    for r in out.itertuples():
        print(f"  chr {r.chromosome} ({r.paralog}, conservation {r.recognition_conservation}): "
              f"{r.recommendation}")
    print("\nSaved data/processed/anchored_predictions.csv")


if __name__ == '__main__':
    main()
