"""Working with DNA motifs: searching for them, discovering them, and describing them.

A "motif" is a short DNA pattern that a protein recognizes and binds to. The four
C. elegans proteins we study each bind a slightly different motif, but all four
share a common frame: they start with the letters TTGG and have a TG a few letters
later. What differs between them is the short "spacer" of letters in between.

A motif is often described not as a single fixed string but as a PWM (Position
Weight Matrix): a small table with one row per DNA letter (A, C, G, T) and one
column per position in the motif. Each column lists how likely each letter is at
that position, so the four numbers in a column add up to 1. We store a PWM as a
4-row grid of numbers.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

# We always order the four DNA letters this way (A, C, G, T) in our tables.
BASES = 'ACGT'
_BASE_INDEX = {b: i for i, b in enumerate(BASES)}

# The known C. elegans motifs, from Phillips et al. 2009 and Li et al. 2024.
KNOWN_MOTIFS = {
    'HIM-8': 'TTGGTCAGTGCA',
    'ZIM-1': 'TTGGTCTGCTA',
    'ZIM-2': 'TTGGGCGCTGCT',
    'ZIM-3': 'TTGGTTGAGGCT',
}

# Each DNA letter pairs with one other letter on the opposite strand.
_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence.

    DNA is double-stranded, and a motif can appear on either strand. The second
    strand reads as this sequence flipped back-to-front with each letter swapped
    for its partner (A with T, G with C). Any unrecognized letter becomes N.
    """
    return ''.join(_COMPLEMENT.get(b, 'N') for b in reversed(seq.upper()))


def scan_motif(sequence, motif):
    """Find every place a motif appears in a longer DNA sequence.

    We look for the motif on both strands (the motif itself and its reverse
    complement). Returns a list, one entry per match, recording the position and
    which strand ("+" or "-") it was found on.
    """
    seq_upper = sequence.upper()
    motif_upper = motif.upper()
    rc_motif = reverse_complement(motif_upper)

    hits = []
    for strand, pattern in [('+', motif_upper), ('-', rc_motif)]:
        start = 0
        while True:
            idx = seq_upper.find(pattern, start)
            if idx == -1:
                break
            hits.append({'position': idx, 'strand': strand})
            start = idx + 1
    return hits


def scan_genome(genome_dict, motifs_dict, species_name, min_length=1_000_000):
    """Search a whole genome for a set of motifs.

    Runs scan_motif for every motif on every full-sized chromosome (short
    fragments are skipped) and collects all the matches into one table.
    """
    all_hits = []
    for chrom, seq in genome_dict.items():
        if len(seq) < min_length:
            continue  # skip short fragments; only search full chromosomes
        for paralog, motif in motifs_dict.items():
            for hit in scan_motif(seq, motif):
                all_hits.append({
                    'species': species_name,
                    'chromosome': chrom,
                    'paralog': paralog,
                    'motif': motif,
                    'position': hit['position'],
                    'strand': hit['strand'],
                })
    return pd.DataFrame(all_hits)


def canonical_kmer(kmer):
    """Pick one standard form for a short DNA piece and its reverse complement.

    A "k-mer" is just a stretch of DNA of length k. Since a piece and its
    reverse complement are the same thing on opposite strands, we always report
    whichever of the two comes first alphabetically, so both count together.
    """
    return min(kmer, reverse_complement(kmer))


def count_kmers(sequence, k):
    """Count how often each short DNA piece (of length k) appears in a sequence.

    Pieces containing an unknown base (N) are skipped.
    """
    counts = Counter()
    seq = sequence.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if 'N' in kmer:
            continue
        counts[canonical_kmer(kmer)] += 1
    return counts


def terminal_enrichment(genome_dict, k=12, terminal_size=2_000_000, min_count=10):
    """Find short DNA pieces that are unusually common at the chromosome ends.

    This is a way to discover candidate motifs without any model: if a short
    sequence appears far more often near the chromosome ends (where pairing
    centers live) than it does across the genome as a whole, it is a candidate
    for the pairing-center motif in that species. For each piece we report how
    many times more common it is at the ends ("fold_enrichment"), and whether it
    contains the TTGG hallmark of this protein family.
    """
    # First, count how common every k-mer is across the whole genome (the
    # baseline we compare the chromosome ends against).
    print("  Computing genome-wide k-mer background...")
    bg_counts = Counter()
    for seq in genome_dict.values():
        if len(seq) < 1_000_000:
            continue
        bg_counts.update(count_kmers(seq, k))
    bg_total = sum(bg_counts.values())

    results = []
    for chrom, seq in genome_dict.items():
        if len(seq) < 2 * terminal_size:
            continue

        # Look at the left end and the right end of each chromosome separately.
        for end_name, region in [('left', seq[:terminal_size]),
                                 ('right', seq[-terminal_size:])]:
            term_counts = count_kmers(region, k)
            term_total = sum(term_counts.values())

            for kmer, count in term_counts.items():
                if count < min_count:
                    continue

                # How common this piece is at the end vs. across the genome.
                term_freq = count / term_total
                bg_freq = bg_counts.get(kmer, 0) / bg_total
                if bg_freq == 0:
                    continue

                results.append({
                    'chromosome': chrom,
                    'end': end_name,
                    'kmer': kmer,
                    'terminal_count': count,
                    'fold_enrichment': term_freq / bg_freq,
                    'starts_with_TTGG': kmer.startswith('TTGG')
                                        or reverse_complement(kmer).startswith('TTGG'),
                })

    return pd.DataFrame(results)


# Turning SELEX experiments into PWMs.
#
# SELEX is a lab experiment that collects many short DNA pieces that a protein
# was found to bind. The 2009 paper (Phillips et al., Table S3) lists these
# pieces one per line, indented with spaces so that matching positions line up
# in the same column, like a hand-aligned block of text. We trust that
# alignment as-is rather than re-aligning the pieces ourselves, because the
# HIM-8 C-terminal site has no TTGG to align on, and re-aligning would also
# throw away the pieces that don't contain an exact TTGG.

def read_selex(path):
    """Read one SELEX file of bound DNA pieces, keeping their alignment.

    Each line holds one DNA piece, indented so matching positions share a
    column. We record each piece together with how far it is indented (its
    starting column). Occasionally a line holds a second piece; those extras
    have no reliable column, so we count and skip them, and report how many.
    """
    rows = []
    n_skipped = 0
    with open(path) as f:
        for line in f:
            matches = list(re.finditer(r'[ACGTNacgtn]+', line))
            if not matches:
                continue
            first = matches[0]
            rows.append((first.start(), first.group().upper()))
            n_skipped += len(matches) - 1  # extra pieces on the same line
    return rows, n_skipped


def alignment_to_counts(rows):
    """Tally, for each aligned column, how many times each DNA letter appears.

    Takes the aligned pieces (each with its starting column) and returns a grid
    of counts (one row per letter A/C/G/T, one column per position) plus how
    many pieces covered each column. Unknown letters are ignored.
    """
    width = max(off + len(s) for off, s in rows)
    counts = np.zeros((4, width), dtype=int)
    for off, seq in rows:
        for i, base in enumerate(seq):
            idx = _BASE_INDEX.get(base)
            if idx is not None:
                counts[idx, off + i] += 1
    coverage = counts.sum(axis=0)
    return counts, coverage


def build_pwm(rows, min_coverage=0.5, pseudocount=0.25):
    """Turn the aligned SELEX pieces into a PWM (the motif's probability table).

    Columns at the ragged edges, seen in fewer than min_coverage of the pieces,
    are dropped. A small "pseudocount" is added to every count before turning
    them into probabilities, so that no letter ends up with a probability of
    exactly zero (which would wrongly say a letter is impossible).

    Returns everything about the motif: the PWM itself, the raw counts, how many
    pieces covered each column, the single most likely letter at each position
    (the "consensus"), and bookkeeping about which columns were kept.
    """
    counts, coverage = alignment_to_counts(rows)
    n_seq = len(rows)

    kept = [j for j in range(counts.shape[1]) if coverage[j] >= min_coverage * n_seq]
    counts = counts[:, kept]
    coverage = coverage[kept]

    smoothed = counts + pseudocount
    pwm = smoothed / smoothed.sum(axis=0, keepdims=True)
    consensus = ''.join(BASES[c] for c in counts.argmax(axis=0))

    return {
        'pwm': pwm,
        'counts': counts,
        'coverage': coverage,
        'consensus': consensus,
        'bases': BASES,
        'n_sequences': n_seq,
        'kept_columns': kept,
    }


def selex_to_pwm(path, min_coverage=0.5, pseudocount=0.25):
    """Read a SELEX file and build its PWM in one step.

    Returns the PWM together with the number of extra same-line pieces that were
    skipped (see read_selex).
    """
    rows, n_skipped = read_selex(path)
    return build_pwm(rows, min_coverage=min_coverage, pseudocount=pseudocount), n_skipped


def motif_to_pwm(motif, pseudocount=0.01):
    """Turn a plain motif string into a PWM.

    This makes a PWM that is almost certain about each position (it puts nearly
    all the probability on the letter actually written in the motif). We use it
    to express the known motifs as PWMs so they can be compared on equal footing
    with predicted ones.
    """
    pwm = np.full((4, len(motif)), pseudocount)
    for j, base in enumerate(motif.upper()):
        idx = _BASE_INDEX.get(base)
        if idx is not None:
            pwm[idx, j] += 1.0
    return pwm / pwm.sum(axis=0, keepdims=True)


# Aligning motifs across proteins by their shared anchors.
#
# Every motif in this family has the same skeleton: TTGG at the 5' end, then a
# short spacer, then a TG "registration" sub-site, then a tail. Only the spacer
# LENGTH and the variable letters differ between proteins. To carry one protein's
# motif over to another protein's frame (needed by the nearest-neighbour and
# learned baselines, which predict one protein's motif from another's), we line
# the two up on BOTH the TTGG and TG anchors rather than naively from the left.

def motif_frame(spacer, motif_len):
    """Return the index lists for the four parts of an anchored motif.

    Given the spacer length and total motif length, report which columns are the
    shared TTGG, the variable spacer, the shared TG, and the variable tail.
    """
    tg = 4 + spacer
    return {
        'ttgg': [0, 1, 2, 3],
        'spacer': list(range(4, 4 + spacer)),
        'tg': [tg, tg + 1],
        'tail': list(range(tg + 2, motif_len)),
    }


def anchored_transfer(src_pwm, src_spacer, dst_len, dst_spacer):
    """Re-express a source motif's PWM in a destination motif's frame.

    Both motifs are registered on their shared anchors: the TTGG at the start and
    the TG sub-site. Because TG is the binding "registration" point, the variable
    spacer between TTGG and TG is aligned to the TG side (right-aligned) and the
    tail after TG is aligned to the TG side (left-aligned). Where the source has
    no column for a destination position (the spacers differ in length), that
    position is left uninformative (each letter equally likely).

    Returns a PWM shaped for the destination motif; used to compare a motif
    predicted for one protein against the known motif of another.
    """
    src = motif_frame(src_spacer, src_pwm.shape[1])
    dst = motif_frame(dst_spacer, dst_len)
    out = np.full((4, dst_len), 0.25)
    # The shared anchors carry straight across.
    for j_out, j_src in zip(dst['ttgg'], src['ttgg']):
        out[:, j_out] = src_pwm[:, j_src]
    for j_out, j_src in zip(dst['tg'], src['tg']):
        out[:, j_out] = src_pwm[:, j_src]
    # Spacer: registered at TG, so fill from the TG side outward (right-aligned).
    n = min(len(src['spacer']), len(dst['spacer']))
    for k in range(1, n + 1):
        out[:, dst['spacer'][-k]] = src_pwm[:, src['spacer'][-k]]
    # Tail: registered at TG, so fill from the TG side outward (left-aligned).
    n = min(len(src['tail']), len(dst['tail']))
    for k in range(n):
        out[:, dst['tail'][k]] = src_pwm[:, src['tail'][k]]
    return out


# A single shared "canonical" frame for the variable positions of every motif,
# so that one learned model can serve all four proteins. The idea is that the
# biologically equivalent positions should share a slot: the widest spacer here
# is 4 letters (HIM-8, ZIM-2) and the longest tail is 5 (ZIM-3), and everything
# is registered on the TG sub-site. A protein's spacer occupies the slots nearest
# TG (right-aligned into the 4 spacer slots) and its tail the slots nearest TG
# (left-aligned into the 5 tail slots); the slots it doesn't use are marked empty.
CANON_SPACER = 4
CANON_TAIL = 5
CANON_VARIABLE = CANON_SPACER + CANON_TAIL  # 9 slots total


def canonical_slots(spacer, motif_len):
    """Map a motif's own variable columns onto the shared canonical slots.

    Returns a list of (canonical_slot_index, motif_column_index) pairs. The
    canonical slots are numbered 0..3 for the four spacer slots (slot 3 is the
    one just before TG) and 4..8 for the five tail slots (slot 4 is just after
    TG). A protein only fills some of them.
    """
    f = motif_frame(spacer, motif_len)
    pairs = []
    # Spacer: right-align into the 4 spacer slots (closest to TG first).
    for k, col in enumerate(reversed(f['spacer'])):
        pairs.append((CANON_SPACER - 1 - k, col))
    # Tail: left-align into the 5 tail slots (closest to TG first).
    for k, col in enumerate(f['tail'][:CANON_TAIL]):
        pairs.append((CANON_SPACER + k, col))
    return pairs


def pwm_to_canonical(pwm, spacer):
    """Lay a motif's variable columns into the 9 shared slots.

    Returns a (9, 4) array of letter probabilities and a length-9 True/False mask
    marking which slots this protein actually uses. Unused slots are left uniform.
    """
    canon = np.full((CANON_VARIABLE, 4), 0.25)
    mask = np.zeros(CANON_VARIABLE, dtype=bool)
    for slot, col in canonical_slots(spacer, pwm.shape[1]):
        canon[slot] = pwm[:, col]
        mask[slot] = True
    return canon, mask


def canonical_to_pwm(canon, spacer, motif_len, anchor_motif=None):
    """Rebuild a motif PWM from its filled canonical slots (inverse of above).

    Places each used canonical slot back into the protein's own motif columns and
    fills the shared TTGG/TG anchors (from anchor_motif if given, else uniform).
    """
    pwm = np.full((4, motif_len), 0.25)
    if anchor_motif is not None:
        f = motif_frame(spacer, motif_len)
        for j in f['ttgg'] + f['tg']:
            col = np.full(4, 0.01)
            col[_BASE_INDEX[anchor_motif[j]]] += 1.0
            pwm[:, j] = col / col.sum()
    for slot, col in canonical_slots(spacer, motif_len):
        pwm[:, col] = canon[slot]
    return pwm
