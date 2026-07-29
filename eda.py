"""First look at the data for the ZIM/HIM-8 pairing-center project.

Run this from the main project folder with: python eda.py

It walks through the whole exploratory analysis: it loads the genomes and gene
annotations, reports basic statistics, finds the four proteins of interest,
searches both genomes for the known motifs, and looks for candidate new motifs in
C. briggsae. The reusable helper functions live in the src/ folder; this script
ties them together and saves its figures and tables into data/processed/.
"""
import numpy as np
from src.paths import P
import pandas as pd
import matplotlib.pyplot as plt

from src.data_io import read_fasta, read_gff, read_fasta_headers
from src.genome_qc import genome_stats, end_quality
from src.orthologs import find_genes, find_proteins, compute_pairwise_identity
from src.motifs import KNOWN_MOTIFS, reverse_complement, scan_genome, terminal_enrichment

## 1. load data
# data for c briggsae
c_briggsae_annotations = read_gff(
    'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.annotations.gff3.gz')
c_briggsae_genome = read_fasta(
    'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.genomic.fa.gz')
c_briggsae_proteins = read_fasta(
    'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.protein.fa.gz')

# data for c elegans
c_elegans_annotations = read_gff(
    'data/raw/c_elegans/caenorhabditis_elegans.PRJNA13758.WBPS19.annotations.gff3.gz')
c_elegans_genome = read_fasta(
    'data/raw/c_elegans/caenorhabditis_elegans.PRJNA13758.WBPS19.genomic.fa.gz')
c_elegans_proteins = read_fasta(
    'data/raw/c_elegans/caenorhabditis_elegans.PRJNA13758.WBPS19.protein.fa.gz')

# Protein-header descriptions (locus names live here, not in the sequence ids).
c_briggsae_protein_headers = read_fasta_headers(
    'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.protein.fa.gz')
c_elegans_protein_headers = read_fasta_headers(
    'data/raw/c_elegans/caenorhabditis_elegans.PRJNA13758.WBPS19.protein.fa.gz')

## 2. basic genome statistics
ce_stats = genome_stats(c_elegans_genome, 'C. elegans')
cb_stats = genome_stats(c_briggsae_genome, 'C. briggsae')

print("=== C. elegans genome ===")
print(ce_stats.to_string(index=False))
print(f"\nTotal: {ce_stats['length'].sum():,} bp across {len(ce_stats)} sequences")

print("\n=== C. briggsae genome ===")
print(cb_stats.to_string(index=False))
print(f"\nTotal: {cb_stats['length'].sum():,} bp across {len(cb_stats)} sequences")

# plot chromosome lengths side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, stats_df, title in [
    (axes[0], ce_stats[ce_stats['length'] > 1_000_000], 'C. elegans chromosomes'),
    (axes[1], cb_stats[cb_stats['length'] > 1_000_000], 'C. briggsae chromosomes')
]:
    ax.barh(stats_df['chromosome'], stats_df['length'] / 1e6, color='steelblue')
    ax.set_xlabel('Length (Mb)')
    ax.set_title(title)
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(P('chromosome_lengths.png'), dpi=150)
# plt.show()


## 3. check for pairing center sequence quality at chromosome ends
ce_ends = end_quality(c_elegans_genome, species_name='C. elegans')
cb_ends = end_quality(c_briggsae_genome, species_name='C. briggsae')

print("=== Chromosome end quality (% N in first/last 200kb) ===")
print("\nC. elegans:")
print(ce_ends[['chromosome', 'length', 'left_n_pct', 'right_n_pct']].to_string(index=False))
print("\nC. briggsae:")
print(cb_ends[['chromosome', 'length', 'left_n_pct', 'right_n_pct']].to_string(index=False))

# flag problematic ends
threshold = 10.0  # more than 10% N is concerning
problematic = pd.concat([ce_ends, cb_ends])
problematic = problematic[
    (problematic['left_n_pct'] > threshold) |
    (problematic['right_n_pct'] > threshold)
]
if len(problematic) > 0:
    print(f"\n  Potentially problematic chromosome ends (>{threshold}% N):")
    print(problematic.to_string(index=False))
else:
    print(f"\n✓ All chromosome ends look clean (<{threshold}% N)")

## 4. annotation overview
print("=== C. elegans annotation feature types ===")
print(c_elegans_annotations['type'].value_counts().head(20))

print("\n=== C. briggsae annotation feature types ===")
print(c_briggsae_annotations['type'].value_counts().head(20))

# check seqid consistency between FASTA and GFF
print("\n=== Chromosome name consistency check ===")
ce_fasta_ids = set(c_elegans_genome.keys())
ce_gff_ids = set(c_elegans_annotations['seqid'].unique())
print(f"C. elegans - In FASTA not GFF: {ce_fasta_ids - ce_gff_ids}")
print(f"C. elegans - In GFF not FASTA: {ce_gff_ids - ce_fasta_ids}")

cb_fasta_ids = set(c_briggsae_genome.keys())
cb_gff_ids = set(c_briggsae_annotations['seqid'].unique())
print(f"C. briggsae - In FASTA not GFF: {cb_fasta_ids - cb_gff_ids}")
print(f"C. briggsae - In GFF not FASTA: {cb_gff_ids - cb_fasta_ids}")

# gene count per chromosome
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, ann_df, title in [
    (axes[0], c_elegans_annotations, 'C. elegans'),
    (axes[1], c_briggsae_annotations, 'C. briggsae')
]:
    gene_counts = (ann_df[ann_df['type'] == 'gene']
                   .groupby('seqid')
                   .size()
                   .sort_values(ascending=False)
                   .head(10))
    ax.barh(gene_counts.index, gene_counts.values, color='coral')
    ax.set_xlabel('Number of genes')
    ax.set_title(f'{title} - genes per chromosome')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(P('gene_counts.png'), dpi=150)
# plt.show()

## 5. finding HIM-8/ZIM orthologs
# C. elegans - straightforward names
ce_search_terms = ['him-8', 'zim-1', 'zim-2', 'zim-3',
                   'WBGene00001872',  # him-8 WormBase ID
                   'WBGene00006992',  # zim-1
                   'WBGene00006993',  # zim-2
                   'WBGene00006994']  # zim-3

ce_zim_genes = find_genes(c_elegans_annotations, ce_search_terms)
print("=== C. elegans ZIM/HIM-8 genes ===")
print(ce_zim_genes[['seqid', 'type', 'start', 'end', 'strand', 'attributes']].to_string(index=False))

# C. briggsae - try Cbr- prefixed names
cb_search_terms = ['Cbr-him-8', 'Cbr-zim-1', 'Cbr-zim-2', 'Cbr-zim-3',
                   'cbr-him-8', 'cbr-zim-1', 'cbr-zim-2', 'cbr-zim-3']

cb_zim_genes = find_genes(c_briggsae_annotations, cb_search_terms)
print("\n=== C. briggsae ZIM/HIM-8 genes (by name) ===")
print(cb_zim_genes[['seqid', 'type', 'start', 'end', 'strand', 'attributes']].to_string(index=False))

if len(cb_zim_genes) == 0:
    print("No hits by name. Try BLAST or broader search.")
    cb_zim_broad = find_genes(c_briggsae_annotations, ['him', 'zim'])
    print("\nBroader search results:")
    print(cb_zim_broad[['seqid', 'type', 'start', 'end', 'attributes']].head(20).to_string(index=False))

## 6. extract protein sequences and compare ZIM/HIM-8 proteins
# Find proteins by name (searching the header descriptions, where loci live).
ce_zim_proteins = find_proteins(
    c_elegans_proteins,
    ['locus=him-8', 'locus=zim-1', 'locus=zim-2', 'locus=zim-3'],
    headers=c_elegans_protein_headers,
)

# In C. briggsae only him-8 is named (Cbr-him-8). The zim genes are not labelled
# by name here, which is the whole reason we later search for them by sequence.
cb_zim_proteins = find_proteins(
    c_briggsae_proteins,
    ['locus=Cbr-him-8', 'locus=Cbr-zim-1', 'locus=Cbr-zim-2', 'locus=Cbr-zim-3'],
    headers=c_briggsae_protein_headers,
)

print("=== C. elegans ZIM/HIM-8 proteins found ===")
for pid, seq in ce_zim_proteins.items():
    print(f"  {pid}: {len(seq)} aa")

print("\n=== C. briggsae ZIM/HIM-8 proteins found ===")
for pid, seq in cb_zim_proteins.items():
    print(f"  {pid}: {len(seq)} aa")

# Check expected lengths (from 2009 paper)
expected_lengths = {'him-8': 361, 'zim-2': 584, 'zim-3': 575}
print("\nExpected lengths from 2009 paper:")
for name, length in expected_lengths.items():
    print(f"  {name}: {length} aa")

## 7. Compute pairwise identity between orthologs
print("\n=== Pairwise Identity (C. elegans vs C. briggsae) ===")
ortholog_pairs = ['him-8', 'zim-1', 'zim-2', 'zim-3']

for name in ortholog_pairs:
    ce_seq = next((seq for pid, seq in ce_zim_proteins.items()
                   if name in c_elegans_protein_headers.get(pid, '')), None)
    cb_seq = next((seq for pid, seq in cb_zim_proteins.items()
                   if name in c_briggsae_protein_headers.get(pid, '')), None)

    if ce_seq and cb_seq:
        identity = compute_pairwise_identity(ce_seq, cb_seq)
        print(f"  {name.upper()}: {identity:.1f}% identity")
    else:
        print(f"  {name.upper()}: One or both sequences missing")

## 8. Scanning for known C. elegans motifs
print("Scanning C. elegans genome for known motifs...")
ce_motif_hits = scan_genome(c_elegans_genome, KNOWN_MOTIFS, 'C. elegans')
print(f"Total hits: {len(ce_motif_hits)}")
print(ce_motif_hits.groupby(['chromosome', 'paralog']).size().unstack(fill_value=0))

ce_motif_hits.to_csv(P('ce_motif_hits.csv'), index=False)


# Plot density along each chromosome
def plot_motif_density(motif_hits, genome_dict, title, bin_size=500_000):
    """Plot motif density along each large chromosome."""
    COLORS = {'HIM-8': 'gold', 'ZIM-1': 'green', 'ZIM-2': 'red', 'ZIM-3': 'orange'}

    main_chroms = sorted([c for c, s in genome_dict.items() if len(s) > 1_000_000])

    fig, axes = plt.subplots(len(main_chroms), 1,
                             figsize=(14, 3 * len(main_chroms)), sharex=False)
    if len(main_chroms) == 1:
        axes = [axes]

    for ax, chrom in zip(axes, main_chroms):
        chrom_len = len(genome_dict[chrom])
        bins = np.arange(0, chrom_len + bin_size, bin_size)

        for paralog, color in COLORS.items():
            hits = motif_hits[(motif_hits['chromosome'] == chrom) &
                              (motif_hits['paralog'] == paralog)]
            if len(hits) == 0:
                continue
            counts, _ = np.histogram(hits['position'], bins=bins)
            ax.bar(bins[:-1] / 1e6, counts, width=bin_size / 1e6,
                   label=paralog, color=color, alpha=0.7)

        ax.set_title(f'Chromosome {chrom}', fontsize=10)
        ax.set_xlabel('Position (Mb)')
        ax.set_ylabel('Motif count\nper 500kb')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


fig = plot_motif_density(
    ce_motif_hits, c_elegans_genome,
    'C. elegans ZIM/HIM-8 motif density (should match Figure 2 of Phillips et al. 2009)')
plt.savefig(P('ce_motif_density.png'), dpi=150, bbox_inches='tight')
# plt.show()

print("""
✓ If this plot shows:
  - HIM-8 (gold) spike at left end of X chromosome
  - ZIM-3 (orange) spikes at left end of I and IV
  - ZIM-1 (green) spikes at left ends of II and III
  - ZIM-2 (red) spike at right end of V
...then your scanning pipeline is working correctly.
""")

## 9. C. briggsae motif density (for comparison)
print("Scanning C. briggsae genome for C. elegans motifs...")
cb_motif_hits = scan_genome(c_briggsae_genome, KNOWN_MOTIFS, 'C. briggsae')
print(f"Total hits: {len(cb_motif_hits)}")
print(cb_motif_hits.groupby(['chromosome', 'paralog']).size().unstack(fill_value=0))

cb_motif_hits.to_csv(P('cb_motif_hits.csv'), index=False)

fig = plot_motif_density(
    cb_motif_hits, c_briggsae_genome,
    'C. briggsae scanned with C. elegans motifs (expect low signal, no clear clustering)')
plt.savefig(P('cb_motif_density_elegans_motifs.png'), dpi=150, bbox_inches='tight')
# plt.show()

# Compare hit counts
print("\n=== Motif hit comparison ===")
comparison = pd.DataFrame({
    'C. elegans hits': ce_motif_hits.groupby('paralog').size(),
    'C. briggsae hits': cb_motif_hits.groupby('paralog').size()
}).fillna(0).astype(int)
print(comparison)
print("\nIf C. briggsae has far fewer hits and no clear clustering,")
print("this confirms the project premise: C. briggsae has different PC motifs.")

## 10. Motif spacing analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Expected spacings from 2009 paper Table S2:
# HIM-8 on X: 21bp tandem; ZIM-3 on IV: 19bp tandem;
# ZIM-1 on II: 63bp total (inverted clusters); ZIM-2 on V: 18bp or 32bp tandem.
spacing_configs = [
    ('HIM-8', 'X', 'HIM-8 on X chromosome (expect ~21bp peak)'),
    ('ZIM-3', 'IV', 'ZIM-3 on chromosome IV (expect ~19bp peak)'),
    ('ZIM-1', 'II', 'ZIM-1 on chromosome II (expect ~63bp total interval)'),
    ('ZIM-2', 'V', 'ZIM-2 on chromosome V (expect 18bp or 32bp peak)'),
]

for ax, (paralog, chrom, title) in zip(axes, spacing_configs):
    hits = (ce_motif_hits[(ce_motif_hits['paralog'] == paralog) &
                          (ce_motif_hits['chromosome'] == chrom)].sort_values('position'))

    if len(hits) < 2:
        ax.text(0.5, 0.5, 'Insufficient hits', transform=ax.transAxes, ha='center')
        ax.set_title(title)
        continue

    spacings = np.diff(hits['position'].values)
    spacings = spacings[spacings < 500]  # care about close clustered motifs

    ax.hist(spacings, bins=50, color='steelblue', edgecolor='white')
    ax.set_xlabel('Spacing between consecutive motif hits (bp)')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=9)

    expected = {'HIM-8': 21, 'ZIM-3': 19, 'ZIM-1': 63, 'ZIM-2': 18}
    if paralog in expected:
        ax.axvline(expected[paralog], color='red', linestyle='--',
                   label=f'Expected: {expected[paralog]}bp')
        ax.legend()

plt.suptitle('Motif spacing distributions (C. elegans)', y=1.02)
plt.tight_layout()
plt.savefig(P('motif_spacings.png'), dpi=150, bbox_inches='tight')
# plt.show()

## 11. k-mer enrichment in C. briggsae terminal regions
print("Computing k-mer enrichment in C. briggsae terminal regions...")
print("(This may take a few minutes...)")
cb_enrichment = terminal_enrichment(c_briggsae_genome, k=12)
cb_enrichment = cb_enrichment.sort_values('fold_enrichment', ascending=False)
cb_enrichment.to_csv(P('cb_kmer_enrichment.csv'), index=False)

print(f"\nTotal enriched k-mers found: {len(cb_enrichment)}")
print("\n=== Top 20 enriched 12-mers per chromosome end ===")
top_per_chrom = (cb_enrichment
                 .groupby(['chromosome', 'end'])
                 .head(5)
                 .sort_values(['chromosome', 'end', 'fold_enrichment'],
                              ascending=[True, True, False]))
print(top_per_chrom[['chromosome', 'end', 'kmer', 'terminal_count',
                     'fold_enrichment', 'starts_with_TTGG']].to_string(index=False))

# Highlight TTGG-containing motifs specifically
ttgg_motifs = cb_enrichment[cb_enrichment['starts_with_TTGG'] == True]
print("\n=== TTGG-containing motifs (biologically motivated filter) ===")
print(f"Found {len(ttgg_motifs)} TTGG-starting k-mers in terminal regions")
print(ttgg_motifs.head(20)[['chromosome', 'end', 'kmer',
                            'terminal_count', 'fold_enrichment']].to_string(index=False))

## 12. Summary
print("""
╔══════════════════════════════════════════════════════════════╗
║                    EDA SUMMARY                               ║
╚══════════════════════════════════════════════════════════════╝
""")

print("── GENOME ASSEMBLY ──────────────────────────────────────────")
print(f"C. elegans:  {len(c_elegans_genome)} sequences, "
      f"{sum(len(s) for s in c_elegans_genome.values()):,} bp total")
print(f"C. briggsae: {len(c_briggsae_genome)} sequences, "
      f"{sum(len(s) for s in c_briggsae_genome.values()):,} bp total")

print("\n── ORTHOLOGS FOUND ──────────────────────────────────────────")
print(f"C. elegans ZIM/HIM-8:  {len(ce_zim_proteins)} proteins found")
print(f"C. briggsae ZIM/HIM-8: {len(cb_zim_proteins)} proteins found")

print("\n── MOTIF SCANNING (C. elegans) ──────────────────────────────")
for paralog in KNOWN_MOTIFS:
    n_hits = len(ce_motif_hits[ce_motif_hits['paralog'] == paralog])
    print(f"  {paralog}: {n_hits} total hits")
print("  → Does this reproduce Figure 2 of Phillips 2009? CHECK PLOT")

print("\n── MOTIF SCANNING (C. briggsae with C. elegans motifs) ──────")
for paralog in KNOWN_MOTIFS:
    n_hits = len(cb_motif_hits[cb_motif_hits['paralog'] == paralog])
    print(f"  {paralog}: {n_hits} total hits (expect far fewer than C. elegans)")

print("\n── CANDIDATE C. BRIGGSAE MOTIFS ─────────────────────────────")
ttgg_count = len(cb_enrichment[cb_enrichment['starts_with_TTGG'] == True])
print(f"  TTGG-containing 12-mers enriched at chromosome ends: {ttgg_count}")
print("  → These are your unsupervised candidate pairing center motifs")
