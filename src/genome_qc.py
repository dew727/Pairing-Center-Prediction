"""Basic quality checks on the genome sequences.

Pairing centers sit at the very ends of chromosomes. Genome assemblies often
have gaps there, shown as runs of the letter "N" (unknown base). If the ends
are full of Ns, the motifs we are looking for would be missing simply because
that stretch of DNA wasn't sequenced well, so we check for this first.
"""
import pandas as pd


def genome_stats(genome_dict, species_name):
    """Summarize each chromosome: its length, how many unknown bases (N) it has,
    and its GC content (the fraction of bases that are G or C).

    Takes the genome as a dictionary of {chromosome name: sequence} and returns
    a table sorted from longest chromosome to shortest.
    """
    stats = []
    for seq_id, seq in genome_dict.items():
        seq_upper = seq.upper()
        length = len(seq)
        n_count = seq_upper.count('N')
        gc_count = seq_upper.count('G') + seq_upper.count('C')
        stats.append({
            'species': species_name,
            'chromosome': seq_id,
            'length': length,
            'n_count': n_count,
            'n_fraction': n_count / length,
            'gc_fraction': gc_count / (length - n_count) if (length - n_count) > 0 else 0,
        })
    return pd.DataFrame(stats).sort_values('length', ascending=False)


def end_quality(genome_dict, window=200_000, species_name=""):
    """Measure the fraction of unknown bases (N) at both ends of each chromosome.

    We look at the first and last stretch of each large chromosome (by default
    the first and last 200,000 bases). A high percentage of Ns there is a
    warning sign that the pairing-center region is poorly assembled. Small
    fragments under 1 million bases are skipped.
    """
    results = []
    for seq_id, seq in genome_dict.items():
        if len(seq) < 1_000_000:
            continue  # skip small fragments; we only care about full chromosomes

        seq_upper = seq.upper()
        left_window = seq_upper[:window]
        right_window = seq_upper[-window:]

        results.append({
            'species': species_name,
            'chromosome': seq_id,
            'length': len(seq),
            'left_n_pct': 100 * left_window.count('N') / window,
            'right_n_pct': 100 * right_window.count('N') / window,
        })

    return pd.DataFrame(results)
