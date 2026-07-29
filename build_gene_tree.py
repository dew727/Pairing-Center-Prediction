"""Build a family tree of these proteins to settle the zim-2 vs zim-3 question.

Simple similarity scores (and BLAST) couldn't decide which C. briggsae gene is
the counterpart of zim-2 versus zim-3. A family tree can help: it groups the
proteins by how they are related, and a protein's closest relative on the tree is
its likely counterpart. We line all the proteins up (with a tool called MAFFT),
build the tree, and repeat the tree-building on 100 resampled versions of the data
to see how trustworthy each grouping is (this repeat-and-count check is called
"bootstrap support").

Run this from the main project folder with: python build_gene_tree.py
It needs the MAFFT program installed and saves three views of the tree into
data/processed/:
  zim_family.nwk            the tree in a standard text format
  zim_family_tree.png       a picture of the tree
  zim_family_tree.txt       a simple text drawing of the tree
"""
import os
from src.paths import P
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Bio import AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Phylo.Consensus import bootstrap_consensus, majority_consensus

from src.data_io import read_fasta
from src.orthologs import make_blosum_aligner, local_align_stats
from find_orthologs import (
    CE_PROTEINS, CB_PROTEINS, CB_GFF, QUERY_LOCI,
    read_headers, representative_queries, gene_coords, best_protein_for_gene,
)

MAFFT_BIN = '/opt/homebrew/bin/mafft'
WORK = os.path.join(tempfile.gettempdir(), 'pc_tree')  # a folder path with no spaces

# The C. briggsae genes we already believe are part of this protein family.
CB_FAMILY = ['CBG12895', 'CBG12897', 'CBG29927', 'CBG12898']
# A few neighbouring genes we throw in to check we haven't missed a family member.
CB_EXTRAS = ['CBG12894', 'CBG27313', 'CBG12899']
N_BOOTSTRAP = 100


def main():
    os.makedirs(WORK, exist_ok=True)
    ce_proteins = read_fasta(CE_PROTEINS)
    cb_proteins = read_fasta(CB_PROTEINS)
    ce_headers = read_headers(CE_PROTEINS)
    cb_headers = read_headers(CB_PROTEINS)
    coords = gene_coords(CB_GFF)
    queries = representative_queries(ce_proteins, ce_headers, QUERY_LOCI)
    seqname_to_gene = {c[5]: g for g, c in coords.items()}

    # Gather the sequences that go into the tree, starting with the four
    # C. elegans proteins. Each gets a short label for the tree.
    records = [(f'Cel_{locus}', seq) for locus, (_, seq, _) in queries.items()]

    aligner = make_blosum_aligner()
    zim3_seq = queries['zim-3'][1]

    def add_cb(seqname, force=False):
        gene = seqname_to_gene.get(seqname)
        pid = best_protein_for_gene(cb_proteins, cb_headers, gene) if gene else None
        if not pid:
            print(f"  (skip {seqname}: no protein)")
            return
        seq = cb_proteins[pid]
        _, ident, cov = local_align_stats(zim3_seq, seq, aligner)
        # For the optional extra genes, only keep ones that actually look like
        # family members (reasonably similar to a known one); drop the rest.
        if not force and (cov < 25 or ident < 18):
            print(f"  (drop {seqname}: not family-like vs zim-3, id={ident:.0f}% cov={cov:.0f}%)")
            return
        records.append((f'Cbr_{seqname}', seq))

    print("Assembling ZIM/HIM-8 family...")
    for sn in CB_FAMILY:
        add_cb(sn, force=True)
    for sn in CB_EXTRAS:
        add_cb(sn)
    print(f"  {len(records)} sequences in the tree: {[r[0] for r in records]}")

    # Line all the sequences up with MAFFT so matching positions are compared.
    fam_fa, aln_fa = f'{WORK}/family.fasta', f'{WORK}/family_aln.fasta'
    with open(fam_fa, 'w') as f:
        for rid, seq in records:
            f.write(f'>{rid}\n{seq}\n')
    print("Aligning with MAFFT...")
    with open(aln_fa, 'w') as out:
        subprocess.run([MAFFT_BIN, '--auto', '--quiet', fam_fa],
                       check=True, stdout=out, text=True)
    aln = AlignIO.read(aln_fa, 'fasta')
    print(f"  alignment: {len(aln)} seqs x {aln.get_alignment_length()} columns")

    # Build the tree from how different the aligned sequences are, then repeat
    # the whole thing on 100 resampled datasets to gauge how reliable it is.
    print(f"Building tree with {N_BOOTSTRAP}x bootstrap support...")
    calc = DistanceCalculator('blosum62')
    constructor = DistanceTreeConstructor(calc, 'nj')
    tree = constructor.build_tree(aln)
    tree.root_at_midpoint()
    for clade in tree.find_clades():          # remove auto-generated internal labels
        if clade.name and clade.name.startswith('Inner'):
            clade.name = None
    consensus = bootstrap_consensus(aln, N_BOOTSTRAP, constructor, majority_consensus)

    Phylo.write(tree, P('zim_family.nwk'), 'newick')
    with open(P('zim_family_tree.txt'), 'w') as f:
        Phylo.draw_ascii(tree, file=f)

    fig, ax = plt.subplots(figsize=(9, 5))
    Phylo.draw(tree, do_show=False, axes=ax)
    ax.set_title('ZIM/HIM-8 family gene tree (Cel = C. elegans, Cbr = C. briggsae)')
    plt.tight_layout()
    plt.savefig(P('zim_family_tree.png'), dpi=150)

    # Read the answer from the shape of the tree (who groups with whom), rather
    # than from raw branch lengths, which can be misleading for this family.
    cb_tips = [r[0] for r in records if r[0].startswith('Cbr_')]
    print("\n" + "=" * 66)
    print("ASCII tree:")
    Phylo.draw_ascii(tree)

    def clade_support(tip_names):
        """How often (out of 100 resamples) exactly this group appeared together."""
        target = set(tip_names)
        for cl in consensus.find_clades():
            if {t.name for t in cl.get_terminals()} == target:
                return cl.confidence
        return None

    def sister_tips(tip):
        """Which proteins are the given protein's closest relatives on the tree."""
        path = tree.get_path(tip)
        parent = path[-2] if len(path) >= 2 else tree.root
        sisters = []
        for child in parent.clades:
            if tip not in [t.name for t in child.get_terminals()]:
                sisters += [t.name for t in child.get_terminals()]
        return sisters

    cel_zims = ['Cel_zim-1', 'Cel_zim-2', 'Cel_zim-3']
    cbr_zims = [t for t in cb_tips if t != 'Cbr_CBG12898']
    s_cel = clade_support(cel_zims)
    s_cbr = clade_support(cbr_zims)
    s_him8 = clade_support(['Cel_him-8', 'Cbr_CBG12898'])

    def fmt(s):
        return f"{s:.0f}/{N_BOOTSTRAP}" if s is not None else "not recovered"

    print("\nOrthology read from tree topology:")
    print(f"  him-8 sister group: {sister_tips('Cel_him-8')}  "
          f"(him-8 + Cbr-him-8 clade support {fmt(s_him8)})")
    print(f"  C. elegans zim-1/2/3 monophyletic? support {fmt(s_cel)}")
    print(f"  C. briggsae zim-cluster monophyletic? support {fmt(s_cbr)}")
    if s_cel and s_cbr:
        print("  => the zim paralogs group BY SPECIES, not 1:1 across species:")
        print("     lineage-specific expansion / concerted evolution. There is no")
        print("     distinct zim-2 vs zim-3 ortholog — the whole zim subfamily is")
        print("     a set of co-orthologs. (him-8 is a clean 1:1 ortholog.)")

    print("\nSaved zim_family.nwk, zim_family_tree.png, zim_family_tree.txt "
          "in data/processed/")


if __name__ == '__main__':
    main()
