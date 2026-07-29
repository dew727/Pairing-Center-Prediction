"""Finding the matching ZIM/HIM-8 genes across species.

Two words used throughout:
  - Orthologs are the "same" gene in two different species, inherited from a
    shared ancestor (for example, C. elegans him-8 and its C. briggsae counterpart).
  - Paralogs are related genes within one species, created by gene duplication
    (him-8, zim-1, zim-2 and zim-3 are paralogs of each other).

C. briggsae only labels its him-8 gene by name, so we have to find the others by
comparing protein sequences and seeing which are most similar.
"""
import re

from Bio import Align
from Bio.Align import substitution_matrices


def find_genes(annotations_df, search_terms, feature_type='gene'):
    """Find genes in an annotation table whose description mentions a search term."""
    mask = annotations_df['type'] == feature_type
    term_mask = annotations_df['attributes'].str.contains(
        '|'.join(search_terms),
        case=False,
        na=False,
    )
    return annotations_df[mask & term_mask].copy()


def find_proteins(proteins_dict, search_terms, headers=None):
    """Find proteins whose name or description mentions one of the search terms.

    By default this only looks at the short protein ID. But gene names such as
    "zim-1" live in the longer description line, not the ID (IDs look like
    "T07G12.6a"). Passing in the descriptions (from read_fasta_headers) lets us
    search those too, which is what actually finds the named genes.
    """
    results = {}
    for prot_id, seq in proteins_dict.items():
        haystack = prot_id if headers is None else f"{prot_id} {headers.get(prot_id, '')}"
        if any(term.lower() in haystack.lower() for term in search_terms):
            results[prot_id] = seq
    return results


def compute_pairwise_identity(seq1, seq2):
    """Measure how similar two protein sequences are, as a percentage.

    We line the two proteins up (an "alignment") and count what fraction of the
    lined-up positions have the exact same amino acid. 100% means identical.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    # Just take the single best alignment. We deliberately avoid asking how many
    # alignments there are: for very different sequences that count can be
    # astronomically large and crash the program.
    try:
        alignment = aligner.align(seq1, seq2)[0]
    except (IndexError, ValueError):
        return 0.0

    # Walk through the lined-up stretches and count exact matches.
    matches = aligned = 0
    for (s1, e1), (s2, e2) in zip(*alignment.aligned):
        aligned += e1 - s1
        matches += sum(a == b for a, b in zip(seq1[s1:e1], seq2[s2:e2]))
    return 100 * matches / aligned if aligned else 0.0


# Searching for orthologs by sequence similarity.
#
# The standard tool for this is BLAST, but it wasn't installed, so we do the
# same thing in plain Python. It works in two steps: a quick rough filter finds
# a handful of promising candidate proteins, then a slower, precise comparison
# scores each candidate. We also do a "reciprocal best hit" check: two proteins
# are called orthologs only if each one is the other's best match, which guards
# against confusing the very similar paralogs.

def parse_fasta_header(header):
    """Break a WormBase protein description line into its labelled parts.

    A description like "CBG12898a wormpep=CBP46385 gene=WBGene00033765
    locus=Cbr-him-8" becomes a dictionary with keys like 'gene' and 'locus', so
    we can look up the gene name or ID easily.
    """
    parts = header.split()
    fields = {'id': parts[0]}
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            fields[key] = value
    return fields


def kmer_set(seq, k=5):
    """Break a protein into all its short overlapping chunks of length k.

    Two related proteins share many of these chunks, so counting the shared ones
    is a fast way to guess how similar two proteins are before doing the slow,
    precise comparison.
    """
    seq = seq.upper()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def make_blosum_aligner(open_gap=-11, extend_gap=-1):
    """Set up the protein comparison tool used for the precise scoring step.

    BLOSUM62 is a standard table that says how interchangeable any two amino
    acids are (some swaps are common and harmless, others are not). The gap
    penalties discourage inserting gaps unless they really improve the match.
    These are the same settings BLAST uses by default.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap
    return aligner


def local_align_stats(query_seq, target_seq, aligner):
    """Compare two proteins and report how well they match.

    Returns three numbers: an overall match score (higher is more similar), the
    percent identity (how many lined-up amino acids are exactly the same), and
    the coverage (how much of the first protein got lined up at all).
    """
    aln = aligner.align(query_seq, target_seq)[0]
    matches = 0
    aligned_cols = 0
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        aligned_cols += qe - qs
        matches += sum(a == b for a, b in zip(query_seq[qs:qe], target_seq[ts:te]))
    identity = 100 * matches / aligned_cols if aligned_cols else 0.0
    coverage = 100 * aligned_cols / len(query_seq) if query_seq else 0.0
    return aln.score, identity, coverage


def build_kmer_index(target_dict, k=5):
    """Pre-compute the short chunks for every protein in a proteome.

    Doing this once up front makes the rough-filter step fast when we later
    compare many query proteins against the same set.
    """
    return {tid: kmer_set(seq, k) for tid, seq in target_dict.items()}


def best_hits(query_seq, target_dict, aligner, k=5, prefilter_n=30, target_kmers=None):
    """Find which proteins in a set are most similar to a given query protein.

    First a fast rough filter (counting shared chunks) keeps the most promising
    candidates; then each candidate is compared precisely and scored. Returns the
    candidates ordered from best match to worst. Passing in target_kmers (from
    build_kmer_index) avoids recomputing the chunks every time.
    """
    if target_kmers is None:
        target_kmers = build_kmer_index(target_dict, k)
    q_kmers = kmer_set(query_seq, k)
    overlaps = []
    for tid in target_dict:
        shared = len(q_kmers & target_kmers[tid])
        if shared:
            overlaps.append((shared, tid))
    overlaps.sort(reverse=True)

    hits = []
    for _, tid in overlaps[:prefilter_n]:
        score, identity, coverage = local_align_stats(query_seq, target_dict[tid], aligner)
        hits.append({'id': tid, 'score': score, 'identity': identity, 'coverage': coverage})
    hits.sort(key=lambda h: h['score'], reverse=True)
    return hits
