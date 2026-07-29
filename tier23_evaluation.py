"""Tier 2/3 biological-plausibility evaluation of the C. briggsae PC motif predictions.

Internal leave-one-paralog-out cannot demonstrate protein->motif generalisation with
only four labelled proteins (see run_baselines.py / train_head.py), and Li et al. 2024
show the specificity determinant (CTD conformation -> spacer length) is not readable
from sequence anyway. So the evidence for our C. briggsae predictions has to come from
biology, not an internal score. This script runs that evaluation.

The C. elegans motifs do NOT occur in C. briggsae at meaningful frequency (the motifs
have co-evolved with the proteins), so we cannot just scan for them. Instead we use the
model-free terminal k-mer enrichment (cb_kmer_enrichment.csv) — the unsupervised
candidate pairing-center motifs enriched at C. briggsae chromosome ends — and ask, for
each chromosome, whether the top candidate behaves like a real pairing-center motif:

  Tier 2a  end clustering      : is it concentrated at a chromosome end?  (by construction)
  Tier 2b  chromosome-specific : is it enriched at ONE chromosome, not genome-wide?
  Tier 2c  tandem structure    : do genomic hits recur with characteristic spacing?
  Tier 2d  TTGG-spacer-TG frame: does it follow the family binding framework?
  Tier 3   convergence         : does it match a C. elegans paralog's motif, and does
                                 that paralog's known target chromosome line up?

Run from the project root:  python tier23_evaluation.py
Writes data/processed/cb_pc_candidates.csv and cb_pc_candidates.png.
"""
from src.paths import P
import numpy as np
import pandas as pd

from src.data_io import read_fasta
from src.motifs import (reverse_complement, scan_motif, KNOWN_MOTIFS,
                        motif_to_pwm, anchored_transfer)
from src.metrics import evaluate, PARALOG_SPACER

KMER_FILE = P('cb_kmer_enrichment.csv')
CB_GENOME = 'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.genomic.fa.gz'

# Known C. elegans chromosome -> binding paralog (Phillips 2009; Li 2024 Fig 1a).
CE_CHROM_PARALOG = {'X': 'HIM-8', 'II': 'ZIM-1', 'III': 'ZIM-1',
                    'V': 'ZIM-2', 'I': 'ZIM-3', 'IV': 'ZIM-3'}


def ttgg_orient(kmer):
    """Return the k-mer in TTGG-first orientation (or None if neither strand has it)."""
    if kmer.startswith('TTGG'):
        return kmer
    rc = reverse_complement(kmer)
    return rc if rc.startswith('TTGG') else None


def is_low_complexity(motif):
    """Flag micro-repeat / low-complexity k-mers (e.g. TTGGCT-TTGGCT) that mimic a
    motif but are really tandem repeats of a short unit — not genuine PC motifs."""
    for unit in (2, 3, 4, 6):
        if len(motif) % unit == 0 and motif == motif[:unit] * (len(motif) // unit):
            return True
    return len(set(motif)) <= 2


def framework_spacer(motif):
    """Infer the spacer length of a TTGG...TG motif, or None if it lacks the frame.

    Looks for the TG registration sub-site 1-4 positions after the TTGG; the gap is
    the spacer length. Returns the smallest valid spacer (TG nearest the TTGG).
    """
    for spacer in (1, 2, 3, 4):
        tg = 4 + spacer
        if motif[tg:tg + 2] == 'TG':
            return spacer
    return None


def best_ce_match(motif, spacer):
    """Which C. elegans paralog's motif is this candidate closest to, and its skill.

    Uses the same locked metric as every baseline: express the candidate in each
    paralog's anchored frame and score it against that paralog's known motif on the
    variable positions. Returns (paralog, skill) for the best-matching paralog.
    """
    cand_pwm = motif_to_pwm(motif)
    best = (None, -np.inf)
    for paralog, ce_motif in KNOWN_MOTIFS.items():
        pred = anchored_transfer(cand_pwm, spacer, len(ce_motif), PARALOG_SPACER[paralog])
        skill = evaluate(pred, motif_to_pwm(ce_motif), paralog)['skill']
        if skill > best[1]:
            best = (paralog, skill)
    return best


def top_candidates(kmers):
    """Pick the single top TTGG-framework candidate motif for each chromosome end."""
    kmers = kmers.copy()
    kmers['motif'] = kmers['kmer'].map(ttgg_orient)
    kmers = kmers.dropna(subset=['motif'])
    kmers['spacer'] = kmers['motif'].map(framework_spacer)
    kmers = kmers.dropna(subset=['spacer'])          # keep only TTGG-spacer-TG frames
    rows = []
    for (chrom, end), g in kmers.groupby(['chromosome', 'end']):
        top = g.sort_values('fold_enrichment', ascending=False).iloc[0]
        rows.append(top)
    return pd.DataFrame(rows), kmers


def chromosome_specificity(motif, kmers):
    """Tier 2b: fraction of this motif's terminal enrichment on its top chromosome.

    1.0 means all its enriched-terminal occurrences are on a single chromosome
    (highly specific, like the C. elegans paralogs); ~1/6 means genome-wide.
    """
    hits = kmers[kmers['motif'] == motif]
    if hits.empty:
        return np.nan, None
    by_chrom = hits.groupby('chromosome')['terminal_count'].sum()
    top_chrom = by_chrom.idxmax()
    return by_chrom.max() / by_chrom.sum(), top_chrom


def tandem_spacing(motif, genome, chrom, end, terminal=2_000_000):
    """Tier 2c: median spacing between consecutive genomic hits at the chromosome end.

    A tandem-repeat pairing center shows regularly-spaced hits; returns the number of
    hits and the median gap between them (bp) in the relevant terminal window.
    """
    seq = genome.get(chrom)
    if seq is None:
        return 0, None
    region = seq[:terminal] if end == 'left' else seq[-terminal:]
    positions = sorted(h['position'] for h in scan_motif(region, motif))
    if len(positions) < 3:
        return len(positions), None
    gaps = np.diff(positions)
    return len(positions), int(np.median(gaps))


def main():
    kmers = pd.read_csv(KMER_FILE)
    kmers = kmers[kmers['starts_with_TTGG']]
    tops, framed = top_candidates(kmers)

    print("Loading C. briggsae genome for the tandem-spacing scan...")
    genome = read_fasta(CB_GENOME)

    rows = []
    for r in tops.itertuples():
        spec, spec_chrom = chromosome_specificity(r.motif, framed)
        n_hits, spacing = tandem_spacing(r.motif, genome, r.chromosome, r.end)
        paralog, skill = best_ce_match(r.motif, int(r.spacer))
        expected = CE_CHROM_PARALOG.get(r.chromosome)
        rows.append({
            'chromosome': r.chromosome, 'end': r.end, 'candidate_motif': r.motif,
            'spacer': int(r.spacer), 'fold_enrichment': round(r.fold_enrichment, 1),
            'terminal_count': int(r.terminal_count),
            'chrom_specificity': round(spec, 2), 'specific_to': spec_chrom,
            'genomic_hits': n_hits, 'median_spacing_bp': spacing,
            'best_ce_paralog': paralog, 'ce_match_skill': round(skill, 2),
            'ce_paralog_for_this_chrom': expected,
            'convergent': paralog == expected,
        })
    out = pd.DataFrame(rows)
    out['low_complexity'] = out['candidate_motif'].map(is_low_complexity)
    # A hit recurring every <15 bp is overlapping/low-complexity, not a real tandem PC.
    out['tandem'] = out['median_spacing_bp'].apply(
        lambda s: bool(s) and 15 <= s <= 5000)
    out = out.sort_values(['chromosome', 'end'])
    out.to_csv(P('cb_pc_candidates.csv'), index=False)

    pd.set_option('display.width', 240, 'display.max_columns', 25)
    print("\nAll top terminal candidates (one per chromosome end):\n")
    show = ['chromosome', 'end', 'candidate_motif', 'spacer', 'fold_enrichment',
            'chrom_specificity', 'genomic_hits', 'median_spacing_bp', 'tandem',
            'low_complexity', 'best_ce_paralog', 'ce_match_skill',
            'ce_paralog_for_this_chrom', 'convergent']
    print(out[show].to_string(index=False))

    # One pairing center per chromosome: keep the best credible candidate per
    # chromosome (drop low-complexity repeats; prefer specific, tandem, enriched).
    cred = out[~out['low_complexity']].copy()
    cred['score'] = (cred['fold_enrichment'] * cred['chrom_specificity']
                     * cred['tandem'].map({True: 1.0, False: 0.4}))
    best = cred.sort_values('score', ascending=False).groupby('chromosome').first()
    best = best.reset_index().sort_values('chromosome')

    print("\n" + "=" * 90)
    print("Best credible pairing-center candidate PER CHROMOSOME:")
    print("=" * 90)
    pcols = ['chromosome', 'end', 'candidate_motif', 'spacer', 'fold_enrichment',
             'chrom_specificity', 'median_spacing_bp', 'best_ce_paralog',
             'ce_paralog_for_this_chrom', 'convergent']
    print(best[pcols].to_string(index=False))

    n_conv = int(best['convergent'].sum())
    n_chrom = len(best)

    # Permutation null: is matching the expected paralog better than chance? Shuffle
    # the chromosome->paralog assignment many times and recompute the match rate.
    rng = np.random.default_rng(0)
    chroms = best['chromosome'].tolist()
    predicted = best['best_ce_paralog'].tolist()
    expected = [CE_CHROM_PARALOG[c] for c in chroms]
    null = []
    for _ in range(20000):
        perm = rng.permutation(list(CE_CHROM_PARALOG.values()))
        mapping = dict(zip(CE_CHROM_PARALOG.keys(), perm))
        null.append(sum(p == mapping[c] for p, c in zip(predicted, chroms)))
    null = np.array(null)
    pval = (null >= n_conv).mean()

    print(f"\nTier 3 convergence: {n_conv}/{n_chrom} chromosomes whose best candidate "
          f"matches the C. elegans paralog known to bind that chromosome.")
    print(f"  permutation null: mean {null.mean():.2f}/{n_chrom} expected by chance; "
          f"P(>= {n_conv}) = {pval:.3f}")
    for r in best[best['convergent']].itertuples():
        print(f"  chr {r.chromosome}: {r.candidate_motif} (spacer {r.spacer}) "
              f"~ {r.best_ce_paralog}  [specificity {r.chrom_specificity}, "
              f"spacing {r.median_spacing_bp} bp]")

    make_figure(best)
    print("\nSaved data/processed/cb_pc_candidates.csv and cb_pc_candidates.png")


def make_figure(best):
    """Bar chart of per-chromosome candidate enrichment, flagged by convergence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    best = best.sort_values('chromosome')
    colors = ['#2c7fb8' if c else '#bdbdbd' for c in best['convergent']]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(best['chromosome'], best['fold_enrichment'], color=colors)
    for bar, r in zip(bars, best.itertuples()):
        tag = f"{r.candidate_motif}\n~{r.best_ce_paralog}" + (" ✓" if r.convergent else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, tag,
                ha='center', va='bottom', fontsize=7)
    ax.set_xlabel('C. briggsae chromosome')
    ax.set_ylabel('terminal fold-enrichment of best candidate')
    ax.set_title('C. briggsae candidate pairing-center motifs (blue = matches the '
                 'expected C. elegans paralog)')
    ax.set_ylim(0, best['fold_enrichment'].max() * 1.25)
    fig.tight_layout()
    fig.savefig(P('cb_pc_candidates.png'), dpi=150)


if __name__ == '__main__':
    main()
