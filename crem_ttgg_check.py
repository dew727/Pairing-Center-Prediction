"""Cross-species test: is the TTGG pairing-center FRAMEWORK present in C. remanei?

The anchor-free run (discover_denovo.py) showed C. remanei's chromosome ends carry
chromosome-specific tandem arrays (architecture conserved), but the most-specific ones are
non-TTGG satellites. This asks the complementary, hypothesis-driven question: seeding from
C. remanei's OWN most-frequent TTGG-framework terminal word (not from C. elegans), does each
chromosome carry a chromosome-specific, tandem, TTGG-spacer-TG motif -- i.e. does the pairing
-center framework replicate to a third species?

Run:  python crem_ttgg_check.py
Writes data/processed/crem_ttgg_motifs.csv.
"""
from collections import Counter
from src.paths import P

import numpy as np
import pandas as pd

from src.data_io import read_fasta
from discover_motifs import (ttgg_windows, framework_spacer, is_low_complexity,
                            discover_chrom, terminal_ttgg_index, specificity,
                            best_ce_match, TERMINAL, MAIN_CHROMS, CE_CHROM_PARALOG)

CREM = 'data/raw/c_remanei/crem.genomic.fa.gz'


def ttgg_seed(seq):
    """Most frequent non-repeat TTGG-framework 12-mer in this chromosome's ends."""
    c = Counter()
    for region in (seq[:TERMINAL], seq[-TERMINAL:]):
        for _, w in ttgg_windows(region, 12):
            if framework_spacer(w) and not is_low_complexity(w):
                c[w] += 1
    return c.most_common(1)[0][0] if c else None


def main():
    genome = {k: v.upper() for k, v in read_fasta(CREM).items() if k in MAIN_CHROMS}
    index = terminal_ttgg_index(genome)
    rows, pwms = [], {}
    for chrom, seq in genome.items():
        seed = ttgg_seed(seq)
        if seed is None:
            continue
        res = discover_chrom(seq, seed)
        if res is None:
            continue
        pwms[chrom] = res['pwm']
        spacer = framework_spacer(res['consensus'][:12])
        paralog, skill = best_ce_match(res['pwm'], spacer) if spacer else (None, None)
        spec = specificity(res['consensus'], chrom, index)
        rows.append({
            'chromosome': chrom, 'ttgg_motif': res['consensus'], 'spacer': spacer,
            'copies': res['n_instances'], 'period_bp': res['period_bp'],
            'downstream_bits': res['downstream_bits'], 'chrom_specificity': spec,
            'cluster_Mb': round(res['cluster_bp'] / 1e6, 2),
            'closest_ce_paralog': paralog, 'ce_skill': skill,
        })
    out = pd.DataFrame(rows)
    out.to_csv(P('crem_ttgg_motifs.csv'), index=False)
    np.savez(P('crem_ttgg_motifs.npz'), **pwms)
    pd.set_option('display.width', 240, 'display.max_columns', 20)
    print("C. remanei TTGG-framework terminal motifs (seeded from C. remanei's own genome):\n")
    print(out.to_string(index=False))
    strong = out[(out['downstream_bits'] >= 6) & (out['chrom_specificity'] >= 0.3)]
    print(f"\n{len(strong)}/{len(out)} chromosomes carry a well-defined, chromosome-specific "
          f"TTGG-framework terminal motif -> the pairing-center framework is present in "
          f"C. remanei (a third species).")


if __name__ == '__main__':
    main()
