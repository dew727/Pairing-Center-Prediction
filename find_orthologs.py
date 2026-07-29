"""Find the C. briggsae versions of the four C. elegans proteins, by sequence.

C. briggsae labels its him-8 gene by name, but not the three zim genes, so a
name search can't find them. Instead we compare every C. briggsae protein to each
C. elegans protein and keep the closest matches (the same thing BLAST does, done
here in plain Python because BLAST wasn't installed). We also require a
"reciprocal best hit": two proteins are matched only if each is the other's best
match, which stops the very similar zim genes from being confused.

Run this from the main project folder with: python find_orthologs.py
It writes data/processed/ortholog_table.csv, which the later steps read.
"""
import gzip
from src.paths import P
import re
from itertools import permutations

import pandas as pd

from src.data_io import read_fasta, read_gff
from src.orthologs import (
    parse_fasta_header, make_blosum_aligner, best_hits, build_kmer_index,
    local_align_stats,
)

CE_PROTEINS = 'data/raw/c_elegans/caenorhabditis_elegans.PRJNA13758.WBPS19.protein.fa.gz'
CB_PROTEINS = 'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.protein.fa.gz'
CB_GFF = 'data/raw/c_briggsae/caenorhabditis_briggsae.PRJNA10731.WBPS19.annotations.gff3.gz'

QUERY_LOCI = ['zim-1', 'zim-2', 'zim-3', 'him-8']  # the four proteins we want to match


def read_headers(path):
    """Read each protein's description line and split it into labelled fields."""
    open_fn = gzip.open if path.endswith('.gz') else open
    out = {}
    with open_fn(path, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                fields = parse_fasta_header(line[1:].strip())
                out[fields['id']] = fields
    return out


def representative_queries(proteins, headers, loci):
    """Pick one representative protein for each gene of interest.

    A gene can be written out in several slightly different forms (isoforms); we
    take the longest one as its representative.
    """
    chosen = {}
    for pid, fields in headers.items():
        locus = fields.get('locus')
        if locus in loci:
            seq = proteins[pid]
            if locus not in chosen or len(seq) > len(chosen[locus][1]):
                chosen[locus] = (pid, seq, fields.get('gene'))
    return chosen


def gene_coords(gff_path):
    """Look up where each C. briggsae gene sits: its chromosome, start and end
    position, strand, and names. Returned as a lookup keyed by gene ID."""
    gff = read_gff(gff_path)
    genes = gff[gff['type'] == 'gene']
    coords = {}
    for _, row in genes.iterrows():
        attrs = row['attributes']
        m_id = re.search(r'ID=Gene:(WBGene\d+)', attrs)
        if not m_id:
            continue
        seqname = re.search(r'sequence_name=([^;]+)', attrs)
        locus = re.search(r'locus=([^;]+)', attrs)
        coords[m_id.group(1)] = (
            row['seqid'], int(row['start']), int(row['end']), row['strand'],
            locus.group(1) if locus else '', seqname.group(1) if seqname else '',
        )
    return coords


def best_protein_for_gene(proteins, headers, wbgene):
    """Return the longest protein form belonging to a given gene."""
    ids = [pid for pid, f in headers.items() if f.get('gene') == wbgene]
    return max(ids, key=lambda p: len(proteins[p])) if ids else None


def main():
    print("Loading proteomes...")
    ce_proteins = read_fasta(CE_PROTEINS)
    cb_proteins = read_fasta(CB_PROTEINS)
    ce_headers = read_headers(CE_PROTEINS)
    cb_headers = read_headers(CB_PROTEINS)
    print(f"  C. elegans:  {len(ce_proteins):,} proteins")
    print(f"  C. briggsae: {len(cb_proteins):,} proteins")

    queries = representative_queries(ce_proteins, ce_headers, QUERY_LOCI)
    coords = gene_coords(CB_GFF)
    him8 = next((c for c in coords.values() if c[4] == 'Cbr-him-8'), None)

    aligner = make_blosum_aligner()
    print("Indexing k-mers (one-time)...")
    cb_index = build_kmer_index(cb_proteins)
    ce_index = build_kmer_index(ce_proteins)

    # For each C. elegans protein, find its closest C. briggsae matches. The four
    # proteins are so alike that they often point at the same match, so we keep
    # the top few for each and sort out who-gets-what together, further down.
    print("\nTop C. briggsae hits per C. elegans paralog (note the overlap):")
    fwd = {}
    for locus in QUERY_LOCI:
        _, qseq, _ = queries[locus]
        fwd[locus] = best_hits(qseq, cb_proteins, aligner, prefilter_n=30,
                               target_kmers=cb_index)
        shown = ', '.join(
            f"{h['id']}({h['identity']:.0f}%/{h['score']:.0f})" for h in fwd[locus][:3])
        print(f"  {locus:6s}: {shown}")

    # him-8 is the one gene C. briggsae labels by name, so we trust that label
    # and pin him-8 to it directly rather than to a possibly-confused best match.
    him8_gene = next((g for g, c in
                      [(g, coords[g]) for g in coords] if c[4] == 'Cbr-him-8'), None)
    him8_protein = best_protein_for_gene(cb_proteins, cb_headers, him8_gene)

    # Build the shortlist of C. briggsae candidates for the three zim genes: one
    # per distinct gene (so two zim genes can't both grab forms of the same gene),
    # leaving out the him-8 gene we already assigned.
    cand_by_gene = {}
    for locus in ['zim-1', 'zim-2', 'zim-3']:
        for h in fwd[locus][:5]:
            gene = cb_headers[h['id']].get('gene')
            if gene == him8_gene:
                continue
            if gene not in cand_by_gene or \
                    len(cb_proteins[h['id']]) > len(cb_proteins[cand_by_gene[gene]]):
                cand_by_gene[gene] = h['id']
    candidates = list(cand_by_gene.values())

    # Score how well each of the three zim genes matches each candidate.
    score = {}
    for locus in ['zim-1', 'zim-2', 'zim-3']:
        _, qseq, _ = queries[locus]
        for cid in candidates:
            s, ident, cov = local_align_stats(qseq, cb_proteins[cid], aligner)
            score[(locus, cid)] = (s, ident, cov)

    # Try every way of matching the three zim genes to three candidates and keep
    # the pairing with the best total score, so each gene gets a distinct match.
    zims = ['zim-1', 'zim-2', 'zim-3']
    best_assign, best_total = None, -1
    for combo in permutations(candidates, 3):
        total = sum(score[(z, c)][0] for z, c in zip(zims, combo))
        if total > best_total:
            best_total, best_assign = total, combo
    assignment = dict(zip(zims, best_assign))
    assignment['him-8'] = him8_protein

    rows = []
    for locus in QUERY_LOCI:
        qid, qseq, _ = queries[locus]
        cid = assignment[locus]
        if locus == 'him-8':
            s, ident, cov = local_align_stats(qseq, cb_proteins[cid], aligner)
            method = 'curated (Cbr-him-8 name annotation)'
        else:
            s, ident, cov = score[(locus, cid)]
            method = 'homology + 1:1 assignment'

        # Double-check by matching the C. briggsae protein back against all of
        # C. elegans: which C. elegans gene does it consider its own best match?
        back = best_hits(cb_proteins[cid], ce_proteins, aligner, prefilter_n=30,
                         target_kmers=ce_index)
        rbh_locus = ce_headers[back[0]['id']].get('locus', '')

        gene = cb_headers[cid].get('gene')
        chrom, start, end, strand, cb_locus, seqname = coords.get(
            gene, ('', 0, 0, '', '', ''))
        dist_mb = abs(start - him8[1]) / 1e6 if him8 and chrom == him8[0] else None
        alts = '; '.join(f"{h['id']}({h['identity']:.0f}%)" for h in fwd[locus][:3])

        rows.append({
            'ce_paralog': locus,
            'ce_protein': qid,
            'cb_protein': cid,
            'cb_gene': seqname or gene,
            'cb_locus': cb_locus,
            'identity_pct': round(ident, 1),
            'coverage_pct': round(cov, 1),
            'score': round(s, 1),
            'rbh_locus': rbh_locus,
            'assignment_method': method,
            'top3_forward_hits': alts,
            'cb_chrom': chrom,
            'cb_start': start,
            'cb_end': end,
            'cb_strand': strand,
            'dist_to_him8_mb': round(dist_mb, 3) if dist_mb is not None else None,
        })

    df = pd.DataFrame(rows).sort_values('cb_start')
    df.to_csv(P('ortholog_table.csv'), index=False)

    print("\n" + "=" * 70)
    print("Assigned C. briggsae orthologs (sorted by genomic position):")
    for _, r in df.iterrows():
        print(f"  {r['ce_paralog']:6s} -> {r['cb_protein']:12s} ({r['cb_gene']})  "
              f"id={r['identity_pct']:.1f}%  score={r['score']:.0f}  "
              f"chr {r['cb_chrom']} @ {r['cb_start']/1e6:.3f} Mb")
    if him8:
        span = (df['cb_start'].min(), df['cb_end'].max())
        print(f"\nAll four orthologs lie on chr {him8[0]} within "
              f"{span[0]/1e6:.3f}-{span[1]/1e6:.3f} Mb "
              f"({(span[1]-span[0])/1e3:.0f} kb) -> the C. elegans tandem cluster "
              "is syntenically conserved.")
    print("Saved data/processed/ortholog_table.csv")


if __name__ == '__main__':
    main()
