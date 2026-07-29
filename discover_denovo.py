"""Drift-safe, architecture-defined pairing-center motif discovery.

The earlier discover_motifs.py hardcodes patterns observed in C. elegans (it only ever
looks at TTGG-anchored windows and assumes the TTGG-spacer-TG framework). That is
circular for predicting in OTHER species: evolutionary drift could have changed the
anchor or the spacer, and a TTGG-constrained search would be structurally blind to it.

This module removes every C. elegans SEQUENCE prior and instead defines a pairing center
only by its drift-ROBUST architecture, which is conserved by mechanism rather than by
sequence:
  * a TANDEM ARRAY of a short repeated unit,
  * near a CHROMOSOME END,
  * CHROMOSOME-SPECIFIC -- each chromosome carries a DIFFERENT unit. This is the crucial
    discriminator: generic telomere/satellite repeats are enriched at ALL ends and shared
    across chromosomes, whereas a pairing center recruits a specific paralog and so its
    motif is specific to one chromosome. So we SELECT candidates by specificity, not by raw
    abundance -- the most abundant terminal repeat is usually NOT the pairing center.

Discovery is DISCRIMINATIVE (STREME-style): a k-mer must be enriched at the ends vs the
chromosome interior, using the TARGET genome's own composition. Among the enriched
candidates we keep the most CHROMOSOME-SPECIFIC one that forms a terminal tandem cluster,
then read the motif (and its width) from the data -- no anchor, length, or spacer assumed.

C. elegans is used only AFTER discovery, to MEASURE drift (did TTGG re-emerge on its own?),
never to constrain it.

NOTE: pure-Python implementation of the discriminative discovery that STREME performs; it
is not the MEME suite. Describe it as "STREME-style discriminative de novo discovery".

Run:  python discover_denovo.py <genome.fa.gz> <tag>
Writes data/processed/denovo_motifs_<tag>.{csv,npz,png}.
"""
import sys
from src.paths import P
from collections import Counter

import numpy as np
import pandas as pd

from src.data_io import read_fasta
from src.motifs import (reverse_complement, canonical_kmer, KNOWN_MOTIFS,
                        motif_to_pwm, anchored_transfer, BASES)
from src.metrics import evaluate, PARALOG_SPACER
from discover_motifs import (is_low_complexity, hamming, MAIN_CHROMS,
                            CE_CHROM_PARALOG, framework_spacer)

K = 12
TERMINAL = 2_000_000
INTERIOR = 2_000_000
CLUSTER_WIN = 50_000
TOP_ENRICHED = 60          # enriched candidates to consider before specificity selection
_BI = {b: i for i, b in enumerate(BASES)}


def count_kmers(seq, k=K):
    c = Counter()
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        if 'N' not in km:
            c[canonical_kmer(km)] += 1
    return c


def terminal_counts(seq):
    c = count_kmers(seq[:TERMINAL])
    c.update(count_kmers(seq[-TERMINAL:]))
    return c


def enriched_candidates(term_c, int_c, top=TOP_ENRICHED):
    """The k-mers most enriched at the ends vs the interior (any sequence, no anchor)."""
    tt, it = sum(term_c.values()), sum(int_c.values())
    scored = []
    for km, n in term_c.items():
        if n < 10 or is_low_complexity(km):
            continue
        enr = (n / tt) / ((int_c.get(km, 0) + 1) / it)
        scored.append((enr, km))
    scored.sort(reverse=True)
    return scored[:top]


def kmer_specificity(km, home, term_counts):
    """Fraction of a k-mer's terminal copies on its home chromosome. High => chromosome-
    specific (pairing-center-like); ~1/6 => generic repeat shared across chromosomes."""
    ck = canonical_kmer(km)
    counts = {c: tc.get(ck, 0) for c, tc in term_counts.items()}
    total = sum(counts.values())
    return (counts.get(home, 0) / total) if total else 0.0


def terminal_cluster(seq, seed):
    """Densest CLUSTER_WIN window of exact seed hits WITHIN a terminal region (either end)."""
    best = None
    for lo, hi in ((0, TERMINAL), (len(seq) - TERMINAL, len(seq))):
        region = seq[lo:hi]
        pos = []
        for m in (seed, reverse_complement(seed)):
            i = region.find(m)
            while i != -1:
                pos.append(lo + i)
                i = region.find(m, i + 1)
        if len(pos) < 5:
            continue
        pos = np.array(sorted(pos))
        for w0 in range(pos.min(), pos.max() - CLUSTER_WIN + 1, 5000):
            n = int(((pos >= w0) & (pos < w0 + CLUSTER_WIN)).sum())
            if best is None or n > best[0]:
                best = (n, w0)
    return best[1] if best else None


def refine(seq, start, seed, max_mm=3):
    """Build + trim the motif PWM from the cluster's near-seed windows. Anchor-free."""
    L = len(seed)
    region = seq[start:start + CLUSTER_WIN]
    inst, pos = [], []
    for i in range(len(region) - L):
        w = region[i:i + L]
        if 'N' in w:
            continue
        if hamming(w, seed) <= max_mm:
            inst.append(w); pos.append(start + i)
        else:
            rc = reverse_complement(w)
            if hamming(rc, seed) <= max_mm:
                inst.append(rc); pos.append(start + i)
    if len(inst) < 8:
        return None
    pwm = _pwm(inst)
    lo = [w for w in inst if _logodds(w, pwm) > L * 0.2] or inst
    pwm = _pwm(lo)
    ic = 2 + np.sum(np.where(pwm > 0, pwm * np.log2(pwm), 0), axis=0)
    keep = np.where(ic >= 0.15)[0]
    core = pwm[:, keep[0]:keep[-1] + 1] if len(keep) else pwm
    consensus = ''.join(BASES[core[:, j].argmax()] for j in range(core.shape[1]))
    period = int(np.median(np.diff(sorted(pos)))) if len(pos) > 2 else None
    bits = float(np.sum(2 + np.sum(np.where(core > 0, core * np.log2(core), 0), axis=0)))
    return {'pwm': core, 'consensus': consensus, 'n_instances': len(lo),
            'cluster_bp': start, 'period_bp': period, 'info_bits': round(bits, 1)}


def _pwm(seqs, pseudo=0.5):
    L = len(seqs[0])
    counts = np.full((4, L), pseudo)
    for s in seqs:
        for j, b in enumerate(s):
            counts[_BI[b], j] += 1
    return counts / counts.sum(axis=0, keepdims=True)


def _logodds(w, pwm):
    return sum(np.log2(pwm[_BI[b], j] / 0.25) for j, b in enumerate(w))


def measure_drift(consensus):
    """AFTER discovery: does TTGG re-emerge unbiased, and which C. elegans motif is nearest?"""
    ttgg = 'TTGG' in consensus or 'TTGG' in reverse_complement(consensus)
    cons = consensus if consensus.startswith('TTGG') else (
        reverse_complement(consensus) if reverse_complement(consensus).startswith('TTGG')
        else consensus)
    spacer = framework_spacer(cons[:12]) if len(cons) >= 12 else None
    best = (None, None)
    if spacer:
        cand = motif_to_pwm(cons[:12])
        bs = -np.inf
        for p, ce in KNOWN_MOTIFS.items():
            s = evaluate(anchored_transfer(cand, spacer, len(ce), PARALOG_SPACER[p]),
                         motif_to_pwm(ce), p)['skill']
            if s > bs:
                bs, best = s, (p, round(s, 2))
    return ttgg, spacer, best[0], best[1]


def main():
    genome_path = sys.argv[1] if len(sys.argv) > 1 else \
        'data/raw/c_briggsae_qx1410/qx1410.genomic.fa.gz'
    tag = sys.argv[2] if len(sys.argv) > 2 else 'qx1410'
    genome = {c: s.upper() for c, s in read_fasta(genome_path).items() if c in MAIN_CHROMS}

    print("Counting terminal + interior k-mers (target genome's own composition)...")
    term_counts, int_counts = {}, {}
    for c, seq in genome.items():
        term_counts[c] = terminal_counts(seq)
        mid = len(seq) // 2
        int_counts[c] = count_kmers(seq[mid - INTERIOR // 2: mid + INTERIOR // 2])

    pwms, rows = {}, []
    for chrom, seq in genome.items():
        cands = enriched_candidates(term_counts[chrom], int_counts[chrom])
        # SELECT by chromosome-specificity (the drift-robust PC discriminator), not abundance.
        best = None
        for enr, seed in cands:
            spec = kmer_specificity(seed, chrom, term_counts)
            start = terminal_cluster(seq, seed)
            if start is None:
                continue
            score = spec                                    # specificity-first selection
            if best is None or score > best[0]:
                best = (score, spec, enr, seed, start)
        if best is None:
            continue
        _, spec, enr, seed, start = best
        res = refine(seq, start, seed)
        if res is None:
            continue
        ttgg, spacer, ce_par, ce_skill = measure_drift(res['consensus'])
        pwms[chrom] = res['pwm']
        rows.append({
            'chromosome': chrom, 'discovered_motif': res['consensus'], 'seed': seed,
            'chrom_specificity': round(spec, 2), 'end_vs_interior_enrichment': enr,
            'copies': res['n_instances'], 'period_bp': res['period_bp'],
            'info_bits': res['info_bits'], 'cluster_Mb': round(res['cluster_bp'] / 1e6, 2),
            'recovered_TTGG': ttgg, 'observed_spacer': spacer,
            'closest_ce_paralog': ce_par, 'ce_similarity_skill': ce_skill,
            'synteny_paralog': CE_CHROM_PARALOG.get(chrom),
        })
    out = pd.DataFrame(rows)
    out.to_csv(P(f'denovo_motifs_{tag}.csv'), index=False)
    np.savez(P(f'denovo_motifs_{tag}.npz'), **pwms)

    pd.set_option('display.width', 260, 'display.max_columns', 30)
    print(f"\nDrift-safe de novo motifs [{tag}] -- selected by chromosome-specificity, "
          f"NO C. elegans sequence prior:\n")
    print(out.to_string(index=False))
    if len(out):
        n = int(out['recovered_TTGG'].sum())
        print(f"\nDRIFT TEST: the unbiased, specificity-selected motif independently "
              f"contains TTGG on {n}/{len(out)} chromosomes.")
        print("  -> where True, the TTGG framework is VALIDATED from data (not imposed);")
        print("  -> where False, it is a candidate genuine divergence to investigate.")


if __name__ == '__main__':
    main()
