"""Discover each C. briggsae chromosome's pairing-center motif directly from the genome.

The protein->motif model can't work (4 examples; the specificity determinant is a CTD
conformation a sequence model can't read). But the C. briggsae genome IS sequenced, and a
pairing center is, by its nature, a dense tandem array of a short motif near a chromosome
end. So rather than predicting the motif from the protein, we find it directly from the
genome, where thousands of copies sit in plain sight.

Per chromosome:
  1. Seed with the most enriched TTGG-framework k-mer at its ends (cb_kmer_enrichment.csv).
  2. Localize the pairing-center cluster: the terminal window where near-seed,
     TTGG-anchored sites pile up (a real PC is a tight tandem array, not the whole end).
  3. Discover the motif: build a PWM from the cluster's TTGG-anchored windows and refine it
     by a few expectation-maximisation passes, so the motif is genuinely found (it can move
     off the exact seed) and its degeneracy + spacer length emerge from the data.
  4. Report the consensus, spacer length, tandem period, instance count, end N-quality
     (pairing centers sit AT the ends, so gappy ends are flagged as unreliable), and the
     synteny-based protein assignment (which C. elegans paralog binds this chromosome).

Run from the project root:  python discover_motifs.py
Writes data/processed/discovered_motifs.{npz,csv,png}.
"""
import numpy as np
from src.paths import P
import pandas as pd

from src.data_io import read_fasta
from src.motifs import (reverse_complement, KNOWN_MOTIFS, motif_to_pwm,
                        anchored_transfer, BASES)
from src.metrics import evaluate, PARALOG_SPACER
from tier23_evaluation import ttgg_orient, framework_spacer, is_low_complexity, CE_CHROM_PARALOG

CB_GENOME = 'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.genomic.fa.gz'
KMER_FILE = P('cb_kmer_enrichment.csv')
MOTIF_LEN = 16          # TTGG + up to 4 spacer + TG + tail
TERMINAL = 3_000_000    # search this much of each chromosome end
CLUSTER_WIN = 50_000    # pairing-center cluster window size
_BI = {b: i for i, b in enumerate(BASES)}


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def ttgg_windows(seq, length=MOTIF_LEN):
    """Yield (position, TTGG-oriented window) for every TTGG on either strand."""
    for i in range(len(seq) - length):
        if seq[i:i + 4] == 'TTGG':
            w = seq[i:i + length]
            if 'N' not in w:
                yield i, w
        elif seq[i:i + 4] == 'CCAA':                      # RC of TTGG (minus strand)
            w = reverse_complement(seq[i - (length - 4):i + 4]) if i >= length - 4 else None
            if w and w.startswith('TTGG') and 'N' not in w:
                yield i, w


def pwm_from(windows, pseudo=0.5):
    counts = np.full((4, MOTIF_LEN), pseudo)
    for w in windows:
        for j, b in enumerate(w):
            counts[_BI[b], j] += 1
    return counts / counts.sum(axis=0, keepdims=True)


def logodds(window, pwm):
    return sum(np.log2(pwm[_BI[b], j] / 0.25) for j, b in enumerate(window))


def locate_cluster(seq, seed):
    """Find the densest terminal window of near-seed TTGG sites; return its windows.

    Localization uses TIGHT seed matching (<=2 mismatches over the 12-bp core): a real
    pairing-center array is many near-identical copies of the motif, so tight matching
    lights up the true tandem array while ignoring the degenerate AT-rich regions that
    would otherwise win a raw-density contest.
    """
    hits = [(i, w) for i, w in ttgg_windows(seq)
            if hamming(w[:len(seed)], seed) <= 2]
    if not hits:
        return None, None
    pos = np.array([h[0] for h in hits])
    best_n, best0 = 0, 0
    for w0 in range(pos.min(), pos.max() - CLUSTER_WIN, 5000):
        n = int(((pos >= w0) & (pos < w0 + CLUSTER_WIN)).sum())
        if n > best_n:
            best_n, best0 = n, w0
    windows = [w for i, w in hits if best0 <= i < best0 + CLUSTER_WIN]
    positions = [i for i, w in hits if best0 <= i < best0 + CLUSTER_WIN]
    return (best0, windows, positions), best_n


def _has_framework(window):
    """True if the window keeps the TTGG-spacer-TG binding framework."""
    return framework_spacer(window[:12]) is not None


def discover(seq, seed, em_rounds=3):
    """Localize the cluster and EM-refine a PWM of the pairing-center motif.

    Refinement is constrained to windows that keep the TTGG-spacer-TG framework, so it
    converges on the real binding motif rather than drifting to the AT-rich / micro-repeat
    sequences that also pile up at chromosome ends. Returns None if no confident,
    framework-following, non-repeat motif survives.
    """
    loc, _ = locate_cluster(seq, seed)
    if loc is None:
        return None
    best0, windows, positions = loc
    # EM candidate pool: framework-following TTGG windows in the cluster only.
    pool = [(i, w) for i, w in ttgg_windows(seq)
            if best0 <= i < best0 + CLUSTER_WIN and _has_framework(w)]
    if len(pool) < 10:
        return None
    all_pos = [i for i, w in pool]
    all_win = [w for i, w in pool]

    seed_win = [w for w in windows if _has_framework(w)] or all_win
    pwm = pwm_from(seed_win)
    inst_pos = positions
    for _ in range(em_rounds):
        scored = sorted(((logodds(w, pwm), w, p)
                         for w, p in zip(all_win, all_pos)), reverse=True)
        keep = [(w, p) for s, w, p in scored if s > MOTIF_LEN * 0.3]
        if len(keep) < 10:
            keep = [(w, p) for _, w, p in scored[:max(20, len(seed_win))]]
        pwm = pwm_from([w for w, p in keep])
        inst_pos = sorted(p for w, p in keep)

    consensus = ''.join(BASES[pwm[:, j].argmax()] for j in range(MOTIF_LEN))
    period = int(np.median(np.diff(inst_pos))) if len(inst_pos) > 2 else None
    # Reject low-complexity / micro-repeat convergence (not a real PC motif).
    if is_low_complexity(consensus[:12]) or (period is not None and period < 15):
        return None
    per_pos = 2 + np.sum(np.where(pwm > 0, pwm * np.log2(pwm), 0), axis=0)
    return {'pwm': pwm, 'consensus': consensus, 'n_instances': len(inst_pos),
            'cluster_start': best0, 'period_bp': period,
            'info_bits': round(float(per_pos.sum()), 1),
            # information in the VARIABLE positions (after the conserved TTGG anchor):
            'downstream_bits': round(float(per_pos[4:].sum()), 1),
            'first_pos': inst_pos[0] if inst_pos else None,
            'last_pos': inst_pos[-1] if inst_pos else None}


def discover_chrom(seq, seed):
    """Discover the pairing-center motif, searching BOTH chromosome ends and keeping
    the stronger cluster. Adds absolute genomic coordinates and distance-from-end."""
    best = None
    for end in ('left', 'right'):
        region = seq[:TERMINAL] if end == 'left' else seq[-TERMINAL:]
        res = discover(region, seed)
        if res is None:
            continue
        offset = 0 if end == 'left' else len(seq) - TERMINAL
        res['end'] = end
        res['cluster_bp'] = offset + res['cluster_start']
        res['dist_from_end_kb'] = round(
            min(res['cluster_bp'], len(seq) - res['cluster_bp']) / 1000)
        # prefer a well-defined, populous cluster
        res['_score'] = res['downstream_bits'] * np.log1p(res['n_instances'])
        if best is None or res['_score'] > best['_score']:
            best = res
    return best


def confidence(res, specific):
    """Tier a prediction: HIGH needs a well-defined downstream motif, tandem
    structure, and a terminal, chromosome-specific cluster."""
    tandem = res['period_bp'] is not None and 20 <= res['period_bp'] <= 3000
    terminal = res['dist_from_end_kb'] <= 1500
    if res['downstream_bits'] >= 6 and tandem and terminal and specific:
        return 'HIGH'
    if res['downstream_bits'] >= 3.5 and tandem:
        return 'MEDIUM'
    return 'LOW'


def n_fraction(seq, window=500_000):
    """N-content of each chromosome end (pairing centers sit here, so it matters)."""
    left = seq[:window].count('N') / window
    right = seq[-window:].count('N') / window
    return round(left, 3), round(right, 3)


def seeds_per_chromosome():
    """Best TTGG-framework k-mer per chromosome (+ which end), to seed discovery."""
    k = pd.read_csv(KMER_FILE)
    k = k[k['starts_with_TTGG']].copy()
    k['motif'] = k['kmer'].map(ttgg_orient)
    k = k.dropna(subset=['motif'])
    k = k[~k['motif'].map(is_low_complexity)]
    k['spacer'] = k['motif'].map(framework_spacer)
    k = k.dropna(subset=['spacer'])
    seeds = {}
    for chrom, g in k.groupby('chromosome'):
        top = g.sort_values('fold_enrichment', ascending=False).iloc[0]
        seeds[chrom] = (top['motif'], top['end'])
    return seeds


def best_ce_match(pwm, spacer):
    """Closest C. elegans paralog to a discovered motif, by the locked metric."""
    best = (None, -np.inf)
    for paralog, ce in KNOWN_MOTIFS.items():
        pred = anchored_transfer(pwm[:, :12], int(spacer), len(ce), PARALOG_SPACER[paralog])
        skill = evaluate(pred, motif_to_pwm(ce), paralog)['skill']
        if skill > best[1]:
            best = (paralog, round(skill, 2))
    return best


MAIN_CHROMS = ['I', 'II', 'III', 'IV', 'V', 'X']


def terminal_ttgg_index(genome):
    """Precompute TTGG 12-mers in every chromosome's terminal regions, so a motif's
    chromosome-specificity can be scored cheaply."""
    idx = {}
    for c in MAIN_CHROMS:
        seq = genome.get(c)
        if seq is None:
            continue
        wins = []
        for region in (seq[:TERMINAL], seq[-TERMINAL:]):
            wins += [w for _, w in ttgg_windows(region, 12)]
        idx[c] = wins
    return idx


def specificity(consensus, home, index):
    """Fraction of a motif's close matches (in all terminal regions) on its home chr.

    A real pairing-center array is many near-identical copies of the motif, so a
    tight threshold (<=1 mismatch over the 12-bp core) counts the array while
    excluding the background TTGG sites scattered elsewhere.
    """
    core = consensus[:12]
    counts = {c: sum(hamming(w, core) <= 1 for w in wins) for c, wins in index.items()}
    total = sum(counts.values())
    return (round(counts.get(home, 0) / total, 2) if total else 0.0)


def main():
    import sys
    genome_path = sys.argv[1] if len(sys.argv) > 1 else CB_GENOME
    tag = sys.argv[2] if len(sys.argv) > 2 else 'af16'
    genome = {c: s.upper() for c, s in read_fasta(genome_path).items()}
    seeds = seeds_per_chromosome()
    index = terminal_ttgg_index(genome)

    pwms, rows = {}, []
    for chrom in MAIN_CHROMS:
        seq = genome.get(chrom)
        if seq is None or chrom not in seeds:
            continue
        seed, _ = seeds[chrom]
        res = discover_chrom(seq, seed)                 # searches BOTH ends
        if res is None:
            continue
        spacer = framework_spacer(res['consensus'][:12]) or framework_spacer(res['consensus'])
        paralog, skill = (best_ce_match(res['pwm'], spacer) if spacer else (None, None))
        spec = specificity(res['consensus'], chrom, index)
        nl, nr = n_fraction(seq)
        n_end = nl if res['end'] == 'left' else nr
        pwms[chrom] = res['pwm']
        rows.append({
            'chromosome': chrom, 'binds_protein': f"Cbr-{paralog}" if paralog else '?',
            'discovered_motif': res['consensus'], 'spacer': spacer,
            'cluster_end': res['end'], 'cluster_pos_Mb': round(res['cluster_bp'] / 1e6, 2),
            'dist_from_end_kb': res['dist_from_end_kb'], 'copies': res['n_instances'],
            'period_bp': res['period_bp'], 'downstream_bits': res['downstream_bits'],
            'chrom_specificity': spec, 'ce_paralog': paralog, 'ce_match_skill': skill,
            'expected_paralog': CE_CHROM_PARALOG.get(chrom),
            'convergent': paralog == CE_CHROM_PARALOG.get(chrom),
            'end_gappy': n_end > 0.05, 'confidence': confidence(res, spec >= 0.5),
        })
    out = pd.DataFrame(rows)
    out.to_csv(P(f'discovered_motifs_{tag}.csv'), index=False)
    np.savez(P(f'discovered_motifs_{tag}.npz'), **pwms)

    pd.set_option('display.width', 260, 'display.max_columns', 30)
    order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    out = out.sort_values('confidence', key=lambda s: s.map(order))
    print(f"De novo C. briggsae pairing-center motif predictions [{tag}]:\n")
    cols = ['chromosome', 'binds_protein', 'discovered_motif', 'spacer', 'cluster_pos_Mb',
            'dist_from_end_kb', 'copies', 'period_bp', 'downstream_bits',
            'chrom_specificity', 'ce_match_skill', 'convergent', 'confidence']
    print(out[cols].to_string(index=False))
    hi = out[out['confidence'] == 'HIGH']
    print(f"\n{len(hi)} HIGH-confidence predictions; "
          f"{int(out['convergent'].sum())}/{len(out)} match the expected paralog.")
    draw_logos(pwms, out, tag)
    print(f"\nSaved data/processed/discovered_motifs_{tag}.{{csv,npz,png}}")


def draw_logos(pwms, out, tag='af16'):
    """Sequence-logo panel of the discovered motif per chromosome."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D

    color = {'A': '#2ca02c', 'C': '#1f77b4', 'G': '#ff7f0e', 'T': '#d62728'}
    chroms = [c for c in ['I', 'II', 'III', 'IV', 'V', 'X'] if c in pwms]
    fig, axes = plt.subplots(len(chroms), 1, figsize=(10, 1.5 * len(chroms)))
    if len(chroms) == 1:
        axes = [axes]
    meta = out.set_index('chromosome')
    for ax, chrom in zip(axes, chroms):
        pwm = pwms[chrom]
        for j in range(pwm.shape[1]):
            ic = 2 + np.sum(np.where(pwm[:, j] > 0, pwm[:, j] * np.log2(pwm[:, j]), 0))
            y = 0
            for b in np.argsort(pwm[:, j]):
                h = pwm[b, j] * ic
                if h < 0.01:
                    continue
                letter = BASES[b]
                tp = TextPath((0, 0), letter, size=1, prop=None)
                bb = tp.get_extents()
                tr = (Affine2D().translate(-bb.x0, -bb.y0)
                      .scale(0.9 / bb.width, h / bb.height)
                      .translate(j + 0.05, y))
                ax.add_patch(PathPatch(tp.transformed(tr), color=color[letter], lw=0))
                y += h
        r = meta.loc[chrom]
        title = f"chr {chrom} ({r.confidence}): {r.discovered_motif}  " \
                f"({r.copies} copies, {r.period_bp} bp period) ~ {r.binds_protein}"
        ax.set_title(title + ("  [gappy end]" if r.end_gappy else ""),
                     fontsize=8, loc='left')
        ax.set_xlim(0, pwm.shape[1]); ax.set_ylim(0, 2)
        ax.set_ylabel('bits', fontsize=7); ax.set_xticks([])
    fig.suptitle(f'De novo C. briggsae pairing-center motifs [{tag}] — discovered from '
                 f'the genome', fontsize=11)
    fig.tight_layout()
    fig.savefig(P(f'discovered_motifs_{tag}.png'), dpi=150)


def reconcile():
    """Combine the two assemblies into one professor-facing prediction sheet.

    Protein assignment is anchored on SYNTENY (chromosome -> paralog, conserved from
    C. elegans), which is more reliable than motif similarity for diverged motifs; motif
    similarity is reported as an independent cross-check. Confidence rewards a well-defined,
    tandem, terminal motif that REPRODUCES across both independently-sequenced strains.
    """
    q = pd.read_csv(P('discovered_motifs_qx1410.csv')).set_index('chromosome')
    a = pd.read_csv(P('discovered_motifs_af16.csv')).set_index('chromosome')
    rows = []
    for c in MAIN_CHROMS:
        if c not in q.index or c not in a.index:
            continue
        rq, ra = q.loc[c], a.loc[c]
        reproduced = hamming(rq.discovered_motif[:12], ra.discovered_motif[:12]) <= 2
        well_defined = (rq.downstream_bits >= 6 and rq.period_bp
                        and 20 <= rq.period_bp <= 3000)
        terminal = rq.dist_from_end_kb <= 1500
        conf = ('HIGH' if well_defined and reproduced and terminal else
                'MEDIUM' if well_defined and (reproduced or terminal) else 'LOW')
        synteny = CE_CHROM_PARALOG.get(c)
        rows.append({
            'chromosome': c, 'binds_protein': f'Cbr-{synteny}',
            'motif': rq.discovered_motif, 'reproduced_on_AF16': reproduced,
            'spacer': rq.spacer, 'cluster_Mb': rq.cluster_pos_Mb, 'copies': rq.copies,
            'period_bp': rq.period_bp, 'downstream_bits': rq.downstream_bits,
            'motif_similar_to': rq.ce_paralog, 'similarity_skill': rq.ce_match_skill,
            'assignment_agrees': rq.ce_paralog == synteny, 'confidence': conf})
    final = pd.DataFrame(rows).sort_values(
        'confidence', key=lambda s: s.map({'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}))
    final.to_csv(P('discovered_motifs_final.csv'), index=False)

    # Redraw the logos with the reconciled synteny assignment + final confidence.
    pwms = dict(np.load(P('discovered_motifs_qx1410.npz')))
    fig_df = final.rename(columns={'motif': 'discovered_motif'}).assign(end_gappy=False)
    draw_logos(pwms, fig_df, 'final')
    pd.set_option('display.width', 260, 'display.max_columns', 30)
    print("FINAL C. briggsae pairing-center motif predictions "
          "(reconciled across QX1410 + AF16):\n")
    print(final.to_string(index=False))
    gold = final[(final.confidence == 'HIGH') & final.assignment_agrees]
    print(f"\nGold-standard (HIGH confidence AND synteny agrees with motif similarity): "
          f"{len(gold)}")
    for r in gold.itertuples():
        print(f"  {r.binds_protein} on chr {r.chromosome}: {r.motif}")
    return final


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--reconcile':
        reconcile()
    else:
        main()
