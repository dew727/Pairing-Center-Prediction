"""Turning extra genomes into extra training labels.

The first model had four labelled proteins because the motif of a ZIM/HIM-8 protein
was only ever measured in *C. elegans* (SELEX in 2009, crystal structures in 2024).
No amount of extra genomes changes that: nobody has run SELEX on *C. remanei*.

What extra genomes do give us is the other half of the pair. This project already
established (RESULTS.md, Layer 6) that a pairing-center motif can be read straight
off a genome, because a pairing center *is* a dense tandem array of its motif near a
chromosome end — many copies sitting in plain sight. So for a new species we can:

    1. find its ZIM/HIM-8 orthologs in the annotated proteome   -> the protein
    2. discover the motif at each chromosome end from the genome -> the motif
    3. join them by synteny (which chromosome each paralog binds) -> a training pair

These are *silver* labels, not gold: they are discovered predictions, so they carry
a confidence and a training weight, and the weakest are dropped rather than trusted.
The four *C. elegans* motifs remain the only gold labels, which is what makes them
the honest held-out test set — they were measured in a lab, not produced by this
pipeline.

Everything here reuses the discovery code that Layer 6 already validated
(`discover_motifs.py`); the only genuinely new part is that the seed k-mers are
computed per species instead of being read from the one C. briggsae file.
"""
import numpy as np
import pandas as pd

from src.motifs import terminal_enrichment
from src.orthologs import build_kmer_index, best_hits, make_blosum_aligner
from src.species import MAIN_CHROMS
from discover_motifs import (discover_chrom, specificity, terminal_ttgg_index,
                             confidence, n_fraction, best_ce_match)
from tier23_evaluation import (ttgg_orient, framework_spacer, is_low_complexity,
                               CE_CHROM_PARALOG)

# The four paralogs, named as the metric names them.
PARALOGS = ['HIM-8', 'ZIM-1', 'ZIM-2', 'ZIM-3']

# How much a label is trusted when the head is trained. A LOW-confidence discovery
# is an unreliable motif; including it at full strength would teach the model noise,
# so by default it is dropped entirely (weight 0).
CONFIDENCE_WEIGHT = {'HIGH': 1.0, 'MEDIUM': 0.4, 'LOW': 0.0}


# ---------------------------------------------------------------------------
# Step 1 — the proteins: find each species' ZIM/HIM-8 orthologs.
# ---------------------------------------------------------------------------

def find_orthologs_in(proteome, queries, min_score=100.0):
    """Match each C. elegans paralog to its counterpart in another species' proteome.

    `queries` maps a paralog name to the C. elegans protein sequence. For each query
    we score the proteome with the same k-mer prefilter plus BLOSUM62 local alignment
    used elsewhere in the project, then assign one protein per paralog.

    The assignment is greedy over the whole score matrix rather than per-query, and no
    protein is used twice. That matters because these four paralogs are recent
    duplicates and are more similar to each other than to anything else, so a
    per-query "best hit" would happily hand the same protein to two paralogs.
    """
    aligner = make_blosum_aligner()
    index = build_kmer_index(proteome)

    scored = []
    for paralog, qseq in queries.items():
        for hit in best_hits(qseq, proteome, aligner, target_kmers=index):
            if hit['score'] >= min_score:
                scored.append((hit['score'], paralog, hit))
    scored.sort(key=lambda t: t[0], reverse=True)

    assigned, used = {}, set()
    for score, paralog, hit in scored:
        if paralog in assigned or hit['id'] in used:
            continue
        assigned[paralog] = {'protein_id': hit['id'], 'score': round(score, 1),
                             'identity': round(hit['identity'], 1),
                             'coverage': round(hit['coverage'], 1)}
        used.add(hit['id'])
    return assigned


# ---------------------------------------------------------------------------
# Step 2 — the motifs: discover each chromosome's pairing-center motif.
# ---------------------------------------------------------------------------

def species_seeds(genome, k=12, terminal_size=2_000_000, min_count=10):
    """Pick the k-mer that seeds motif discovery on each chromosome of one species.

    Same rule the C. briggsae run used, but computed from this species' own genome:
    take the k-mers most over-represented at the chromosome ends relative to the whole
    genome, keep the ones that carry the TTGG-spacer-TG binding framework, drop
    low-complexity micro-repeats, and use the most enriched survivor per chromosome.

    Seeding this way (rather than with a C. elegans motif) is what keeps the search
    from assuming the answer in a species whose motif has drifted.
    """
    # The enrichment background is built only from sequences of at least 1 Mb. A real
    # Caenorhabditis chromosome is 13-21 Mb, so failing this means the assembly is not
    # chromosome-level and should not have reached here — say so rather than dividing
    # by an empty background further down.
    if not any(len(s) >= 1_000_000 for s in genome.values()):
        raise ValueError(
            "No sequence of at least 1 Mb: this assembly is not chromosome-level, so "
            "terminal motif discovery cannot run on it. It should have been excluded "
            "by the annotation gate in prepare_species.py.")

    enr = terminal_enrichment(genome, k=k, terminal_size=terminal_size,
                              min_count=min_count)
    if enr.empty:
        return {}
    enr = enr[enr['starts_with_TTGG']].copy()
    enr['motif'] = enr['kmer'].map(ttgg_orient)
    enr = enr.dropna(subset=['motif'])
    enr = enr[~enr['motif'].map(is_low_complexity)]
    enr['spacer'] = enr['motif'].map(framework_spacer)
    enr = enr.dropna(subset=['spacer'])

    seeds = {}
    for chrom, g in enr.groupby('chromosome'):
        top = g.sort_values('fold_enrichment', ascending=False).iloc[0]
        seeds[chrom] = top['motif']
    return seeds


def discover_species(genome, seeds=None):
    """Discover the pairing-center motif on every chromosome of one species.

    Returns one record per chromosome where a motif was found, carrying the motif,
    its PWM, how tandem and how chromosome-specific it is, and a confidence tier —
    the same fields the validated C. briggsae run produced.
    """
    if seeds is None:
        seeds = species_seeds(genome)
    index = terminal_ttgg_index(genome)

    records = []
    for chrom in MAIN_CHROMS:
        seq = genome.get(chrom)
        if seq is None or chrom not in seeds:
            continue
        res = discover_chrom(seq, seeds[chrom])
        if res is None:
            continue
        consensus = res['consensus']
        spacer = framework_spacer(consensus[:12]) or framework_spacer(consensus)
        if spacer is None:
            continue                       # no binding framework: not a PC motif
        spec = specificity(consensus, chrom, index)
        nl, nr = n_fraction(seq)
        ce_paralog, ce_skill = best_ce_match(res['pwm'], spacer)
        records.append({
            'chromosome': chrom,
            'motif': consensus,
            'pwm': res['pwm'],
            'spacer': int(spacer),
            'copies': res['n_instances'],
            'period_bp': res['period_bp'],
            'downstream_bits': res['downstream_bits'],
            'chrom_specificity': spec,
            'cluster_end': res['end'],
            'dist_from_end_kb': res['dist_from_end_kb'],
            'end_gappy': (nl if res['end'] == 'left' else nr) > 0.05,
            'ce_match_paralog': ce_paralog,
            'ce_match_skill': ce_skill,
            'confidence': confidence(res, spec >= 0.5),
        })
    return records


# ---------------------------------------------------------------------------
# Step 3 — the join: pair each discovered motif with the protein that binds it.
# ---------------------------------------------------------------------------

def build_labels(species_key, species_name, short, orthologs, discoveries,
                 min_weight=0.0):
    """Join discovered motifs to orthologous proteins into silver training labels.

    The join is by synteny: in *C. elegans* each paralog binds a known chromosome
    (HIM-8 the X, ZIM-1 chromosomes II and III, ZIM-2 chromosome V, ZIM-3 chromosomes
    I and IV), and that chromosome-to-paralog assignment is conserved — it is the
    basis Layer 6 already used, and it is more robust than motif similarity for a
    motif that has drifted.

    Two chromosomes map to the same paralog (II/III to ZIM-1, I/IV to ZIM-3), so one
    protein can pick up two labels. Both are kept as separate weighted examples: in
    *C. elegans* the two agree, and where they disagree that genuinely is the
    uncertainty in the label, which the weighting is there to absorb.

    A label carrying no weight (a LOW-confidence discovery) is returned but flagged,
    so the exclusion is visible in the output table rather than silent.
    """
    rows = []
    for d in discoveries:
        paralog = CE_CHROM_PARALOG.get(d['chromosome'])
        if paralog is None or paralog not in orthologs:
            continue
        ortho = orthologs[paralog]
        weight = CONFIDENCE_WEIGHT.get(d['confidence'], 0.0)
        if d['end_gappy']:
            weight *= 0.5          # a gappy end makes the terminal read less reliable
        rows.append({
            'species': species_key,
            'species_name': species_name,
            'short': short,
            'paralog': paralog,
            'chromosome': d['chromosome'],
            'protein_id': ortho['protein_id'],
            'ortholog_identity': ortho['identity'],
            'ortholog_score': ortho['score'],
            'motif': d['motif'],
            'spacer': d['spacer'],
            'copies': d['copies'],
            'period_bp': d['period_bp'],
            'downstream_bits': d['downstream_bits'],
            'chrom_specificity': d['chrom_specificity'],
            'dist_from_end_kb': d['dist_from_end_kb'],
            'end_gappy': d['end_gappy'],
            'ce_match_paralog': d['ce_match_paralog'],
            'ce_match_skill': d['ce_match_skill'],
            'confidence': d['confidence'],
            'weight': round(weight, 2),
            'used_for_training': weight > min_weight,
            'label_type': 'silver',
        })
    return rows


def label_key(row):
    """The name a label's protein carries in the embedding file: 'Cbr_zim-1'."""
    return f"{row['short']}_{row['paralog'].lower()}"


def summarize(labels):
    """A short per-species tally of how many usable labels each genome contributed."""
    if not len(labels):
        return pd.DataFrame()
    df = pd.DataFrame(labels) if not isinstance(labels, pd.DataFrame) else labels
    return (df.groupby('species')
              .agg(labels=('motif', 'size'),
                   used=('used_for_training', 'sum'),
                   high=('confidence', lambda s: int((s == 'HIGH').sum())),
                   mean_weight=('weight', 'mean'))
              .reset_index()
              .sort_values('used', ascending=False))
