"""Side-by-side comparison of the two C. remanei discovery runs.

Run A -- anchor-free, specificity-only (discover_denovo.py): find the MOST chromosome-specific
         terminal tandem repeat, no sequence prior. Picks satellite repeats, no TTGG.
Run B -- TTGG-framework (crem_ttgg_check.py): among TTGG-starting terminal words (seeded from
         C. remanei's OWN genome), find the chromosome-specific tandem motif. Recovers the
         pairing-center motifs, including the ZIM-1 motif conserved across three species.

The contrast is the methodological point: chromosome ends are full of specific satellite repeats,
so "most specific repeat" is NOT the pairing center -- you must look within the TTGG framework
(justified by the conserved binding proteins) to find it.

Run:  python compare_runs.py
Writes data/processed/crem_run_comparison.{csv,png}.
"""
import numpy as np
from src.paths import P
import pandas as pd

from discover_motifs import MAIN_CHROMS
from src.motifs import BASES


def load(csv, npz):
    df = pd.read_csv(csv).set_index('chromosome')
    pw = np.load(npz)
    return df, {k: pw[k] for k in pw.files}


def draw_logo(ax, pwm):
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    color = {'A': '#2ca02c', 'C': '#1f77b4', 'G': '#ff7f0e', 'T': '#d62728'}
    for j in range(pwm.shape[1]):
        ic = 2 + np.sum(np.where(pwm[:, j] > 0, pwm[:, j] * np.log2(pwm[:, j]), 0))
        y = 0
        for b in np.argsort(pwm[:, j]):
            h = pwm[b, j] * ic
            if h < 0.01:
                continue
            tp = TextPath((0, 0), BASES[b], size=1)
            bb = tp.get_extents()
            tr = (Affine2D().translate(-bb.x0, -bb.y0)
                  .scale(0.9 / bb.width, h / bb.height).translate(j + 0.05, y))
            ax.add_patch(PathPatch(tp.transformed(tr), color=color[BASES[b]], lw=0))
            y += h
    ax.set_xlim(0, pwm.shape[1]); ax.set_ylim(0, 2)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    free_df, free_pw = load(P('denovo_motifs_cremanei.csv'),
                            P('denovo_motifs_cremanei.npz'))
    ttgg_df, ttgg_pw = load(P('crem_ttgg_motifs.csv'),
                            P('crem_ttgg_motifs.npz'))

    rows = []
    for c in MAIN_CHROMS:
        f = free_df.loc[c] if c in free_df.index else None
        t = ttgg_df.loc[c] if c in ttgg_df.index else None
        rows.append({
            'chromosome': c,
            'ANCHOR_FREE_motif': f['discovered_motif'] if f is not None else None,
            'ANCHOR_FREE_specificity': f['chrom_specificity'] if f is not None else None,
            'ANCHOR_FREE_has_TTGG': f['recovered_TTGG'] if f is not None else None,
            'TTGG_motif': t['ttgg_motif'] if t is not None else None,
            'TTGG_specificity': t['chrom_specificity'] if t is not None else None,
            'TTGG_closest_paralog': t['closest_ce_paralog'] if t is not None else None,
            'TTGG_match_skill': t['ce_skill'] if t is not None else None,
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(P('crem_run_comparison.csv'), index=False)
    pd.set_option('display.width', 240, 'display.max_columns', 20)
    print("C. remanei -- two runs compared:\n")
    print(comp.to_string(index=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chroms = [c for c in MAIN_CHROMS if c in free_pw and c in ttgg_pw]
    fig, axes = plt.subplots(len(chroms), 2, figsize=(12, 1.5 * len(chroms)))
    for i, c in enumerate(chroms):
        draw_logo(axes[i, 0], free_pw[c])
        f = free_df.loc[c]
        axes[i, 0].set_ylabel(f"chr {c}", rotation=0, labelpad=22, va='center', fontsize=11)
        axes[i, 0].set_title(f"{f['discovered_motif']}  (specificity {f['chrom_specificity']}, "
                             f"no TTGG)", fontsize=8, loc='left')
        draw_logo(axes[i, 1], ttgg_pw[c])
        t = ttgg_df.loc[c]
        tag = f"~{t['closest_ce_paralog']} (match {t['ce_skill']:+.2f})" if pd.notna(t['closest_ce_paralog']) else ""
        axes[i, 1].set_title(f"{t['ttgg_motif']}  {tag}", fontsize=8, loc='left')
    axes[0, 0].annotate("RUN A: anchor-free (most chromosome-specific repeat)",
                        xy=(0.5, 1.5), xycoords='axes fraction', ha='center',
                        fontsize=11, fontweight='bold')
    axes[0, 1].annotate("RUN B: TTGG-framework (pairing-center motif)",
                        xy=(0.5, 1.5), xycoords='axes fraction', ha='center',
                        fontsize=11, fontweight='bold')
    fig.suptitle("C. remanei: 'most specific repeat' (junk satellites) vs the TTGG "
                 "pairing-center framework", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(P('crem_run_comparison.png'), dpi=150, bbox_inches='tight')
    print("\nSaved data/processed/crem_run_comparison.{csv,png}")


if __name__ == '__main__':
    main()
