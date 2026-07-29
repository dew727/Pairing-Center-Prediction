"""Central mapping of every generated artifact to a categorized subfolder.

All scripts write and read their outputs through `P(filename)`, which routes each
file into `data/processed/<category>/` by method/run stage, so the results folder
stays organized and every producer/consumer agrees on the location by construction.
"""
import os

BASE = 'data/processed'

# Ordered (prefix -> subfolder) rules; the first matching prefix wins. Prefixes are
# disjoint under str.startswith, so ordering is just for readability.
_RULES = [
    ('discovered_motifs',   '08_discovery_ttgg'),
    ('denovo_motifs',       '09_discovery_anchorfree'),
    ('anchored_predictions', '10_anchored_reconciliation'),
    ('crem_',               '11_cremanei'),
    ('jaspar_',             '06_jaspar_pretraining'),
    ('embedding',           '04_embeddings'),        # embeddings*, embedding_similarity*
    ('pwms',                '02_selex_pwms'),
    ('ortholog_',           '03_orthologs'),
    ('zim_family',          '03_orthologs'),
    ('baseline_scores',     '05_baselines'),
    ('head_',               '05_baselines'),
    ('cb_predicted_motifs', '05_baselines'),
    ('cb_pc_candidates',    '07_tier23_biology'),
    ('cb_kmer_enrichment',  '01_eda'),
    ('cb_motif',            '01_eda'),
    ('ce_motif',            '01_eda'),
    ('chromosome_lengths',  '01_eda'),
    ('gene_counts',         '01_eda'),
    ('motif_spacings',      '01_eda'),
]


def subdir(filename):
    """The category subfolder for a given output filename (empty string if unknown)."""
    for prefix, sub in _RULES:
        if filename.startswith(prefix):
            return sub
    return ''


def P(filename):
    """Full path for an artifact, creating its category subfolder on demand."""
    d = os.path.join(BASE, subdir(filename))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)
