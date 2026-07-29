"""Build motif tables (PWMs) from the 2009 lab binding data.

Run this from the main project folder with: python build_selex_pwms.py

It reads the SELEX files in data/raw/selex/ (lists of DNA pieces each protein was
found to bind, from Phillips et al. 2009) and turns each one into a PWM: a table
describing the motif that protein prefers. It saves two files:

  - data/processed/pwms.npz       the motif tables and the raw counts behind them
  - data/processed/pwms_meta.json a plain-text summary (the consensus motif, how
                                  many sequences were used, and the settings)

These PWMs are the answers the model will learn to predict from a protein's
sequence. ZIM-1 has no PWM here because no binding experiment was done for it in
2009; its motif will come from the 2024 crystal structure instead.
"""
import json
from src.paths import P

import numpy as np

from src.motifs import selex_to_pwm

# Each SELEX file, with: a short label, which part of the protein it came from,
# and whether it is the pairing-center motif we care about.
SELEX_FILES = {
    'data/raw/selex/him-8-znf.txt':   ('HIM-8',     'ZnF core',      True),
    'data/raw/selex/zim-2-znf.txt':   ('ZIM-2',     'ZnF core',      True),
    'data/raw/selex/zim-3-znf.txt':   ('ZIM-3',     'ZnF core',      True),
    'data/raw/selex/him-8-cterm.txt': ('HIM-8-CTD', 'C-terminus',    False),
}

MIN_COVERAGE = 0.5   # keep alignment columns observed in >= 50% of oligos
PSEUDOCOUNT = 0.25   # added per cell before normalising (no zero probabilities)


def main():
    arrays = {}
    meta = {
        'source': 'Phillips et al. 2009, Nat Cell Biol (DOI 10.1038/ncb1904), Table S3',
        'bases': 'ACGT',
        'pwm_shape': '(4, L): rows A,C,G,T; columns positions; each column sums to 1',
        'min_coverage': MIN_COVERAGE,
        'pseudocount': PSEUDOCOUNT,
        'alignment': 'leading-whitespace alignment from the source file (trusted, not re-anchored)',
        'note_zim1': 'No ZIM-1 SELEX data in 2009; motif from 2024 crystal structure instead.',
        'motifs': {},
    }

    print("Building SELEX PWMs")
    print("=" * 64)
    for path, (label, domain, is_pc) in SELEX_FILES.items():
        result, n_skipped = selex_to_pwm(
            path, min_coverage=MIN_COVERAGE, pseudocount=PSEUDOCOUNT)

        arrays[f'{label}__pwm'] = result['pwm']
        arrays[f'{label}__counts'] = result['counts']
        arrays[f'{label}__coverage'] = result['coverage']

        kept = result['kept_columns']
        meta['motifs'][label] = {
            'source_file': path,
            'domain': domain,
            'is_pairing_center_motif': is_pc,
            'n_sequences': result['n_sequences'],
            'n_secondary_oligos_skipped': n_skipped,
            'width': len(result['consensus']),
            'consensus': result['consensus'],
            'kept_columns': [int(kept[0]), int(kept[-1])],
            'min_per_column_coverage': int(result['coverage'].min()),
        }

        flag = 'pairing-center motif' if is_pc else 'separate C-terminal site'
        print(f"\n{label}  ({domain}, {flag})")
        print(f"  oligos:    {result['n_sequences']}"
              + (f"  (+{n_skipped} secondary skipped)" if n_skipped else ""))
        print(f"  width:     {len(result['consensus'])} bp"
              f"  (alignment cols {kept[0]}..{kept[-1]})")
        print(f"  consensus: {result['consensus']}")

    np.savez(P('pwms.npz'), **arrays)
    with open(P('pwms_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 64)
    print("Saved data/processed/pwms.npz and data/processed/pwms_meta.json")
    print(f"  {len(SELEX_FILES)} PWMs: {', '.join(l for l, _, _ in SELEX_FILES.values())}")
    print("  (ZIM-1 has no SELEX data — see metadata note)")


if __name__ == '__main__':
    main()
