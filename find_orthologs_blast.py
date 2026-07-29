"""Find the C. briggsae orthologs again, this time using the real BLAST tool.

This does the same job as find_orthologs.py (pin him-8 to its named C. briggsae
gene, then match the three zim genes one-to-one), but it uses the standard BLAST
program instead of our hand-written search. Running it both ways is a check: if
the two independent methods agree, we can be more confident in the answer.

Run this from the main project folder with: python find_orthologs_blast.py
It needs the BLAST program installed (see BLAST_BIN below) and writes
data/processed/ortholog_table_blast.csv.

One quirk handled here: BLAST breaks on folder names containing spaces, and this
project's folder has one, so BLAST's working files are kept in a temporary folder
with no spaces in its path. Only the final results file goes back into the project.
"""
import gzip
from src.paths import P
import os
import shutil
import subprocess
import tempfile
from itertools import permutations

import pandas as pd

from src.data_io import read_fasta
from find_orthologs import (
    CE_PROTEINS, CB_PROTEINS, CB_GFF, QUERY_LOCI,
    read_headers, representative_queries, gene_coords, best_protein_for_gene,
)

BLAST_BIN = '/opt/homebrew/bin'
# Folder for BLAST's working files, chosen to have no spaces in its path (BLAST
# can't handle spaces). Only the final results file goes back into the project.
WORK = os.path.join(tempfile.gettempdir(), 'pc_blast')
OUTFMT = '6 qseqid sseqid pident length qcovs bitscore evalue'


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n--- stderr ---\n{proc.stderr}")


def gunzip(src, dst):
    with gzip.open(src, 'rt') as f_in, open(dst, 'w') as f_out:
        shutil.copyfileobj(f_in, f_out)


def write_fasta(path, records):
    with open(path, 'w') as f:
        for rid, seq in records:
            f.write(f'>{rid}\n{seq}\n')


def make_db(fasta, name):
    run([f'{BLAST_BIN}/makeblastdb', '-in', fasta, '-dbtype', 'prot',
         '-out', f'{WORK}/{name}'])


def blastp(query_fasta, db, out_tsv):
    run([f'{BLAST_BIN}/blastp', '-query', query_fasta, '-db', f'{WORK}/{db}',
         '-out', out_tsv, '-outfmt', OUTFMT, '-evalue', '1e-3',
         '-max_target_seqs', '50'])
    cols = ['qseqid', 'sseqid', 'pident', 'length', 'qcovs', 'bitscore', 'evalue']
    df = pd.read_csv(out_tsv, sep='\t', names=cols)
    # If a protein pair matches in several places, keep only their best match.
    return df.sort_values('bitscore', ascending=False).drop_duplicates(['qseqid', 'sseqid'])


def main():
    os.makedirs(WORK, exist_ok=True)
    print("Loading proteomes / metadata...")
    ce_proteins = read_fasta(CE_PROTEINS)
    cb_proteins = read_fasta(CB_PROTEINS)
    ce_headers = read_headers(CE_PROTEINS)
    cb_headers = read_headers(CB_PROTEINS)
    coords = gene_coords(CB_GFF)
    him8 = next((c for c in coords.values() if c[4] == 'Cbr-him-8'), None)
    queries = representative_queries(ce_proteins, ce_headers, QUERY_LOCI)
    qid_to_locus = {qid: locus for locus, (qid, _, _) in queries.items()}

    print("Preparing FASTAs and BLAST databases...")
    cb_fasta, ce_fasta = f'{WORK}/cb_proteins.fasta', f'{WORK}/ce_proteins.fasta'
    gunzip(CB_PROTEINS, cb_fasta)
    gunzip(CE_PROTEINS, ce_fasta)
    make_db(cb_fasta, 'cb')
    make_db(ce_fasta, 'ce')
    write_fasta(f'{WORK}/queries.fasta', [(qid, seq) for qid, seq, _ in queries.values()])

    print("Running forward blastp (C. elegans paralogs vs C. briggsae proteome)...")
    fwd = blastp(f'{WORK}/queries.fasta', 'cb', f'{WORK}/forward.tsv')
    fwd['locus'] = fwd['qseqid'].map(qid_to_locus)

    print("\nTop C. briggsae blastp hits per paralog (sseqid, %id, E-value, bitscore):")
    for locus in QUERY_LOCI:
        sub = fwd[fwd['locus'] == locus].head(3)
        shown = ', '.join(f"{r.sseqid}({r.pident:.0f}%, E={r.evalue:.0e}, b={r.bitscore:.0f})"
                          for r in sub.itertuples())
        print(f"  {locus:6s}: {shown}")

    # Look up each match score by (C. elegans gene, C. briggsae protein).
    bit = {(r.locus, r.sseqid): r for r in fwd.itertuples()}

    # Pin him-8 to the C. briggsae gene that is named after it.
    him8_gene = next((g for g, c in coords.items() if c[4] == 'Cbr-him-8'), None)
    him8_protein = best_protein_for_gene(cb_proteins, cb_headers, him8_gene)

    # Shortlist of candidates for the zim genes: one per gene, leaving out him-8.
    cand_by_gene = {}
    for locus in ['zim-1', 'zim-2', 'zim-3']:
        for r in fwd[fwd['locus'] == locus].itertuples():
            gene = cb_headers[r.sseqid].get('gene')
            if gene == him8_gene:
                continue
            if gene not in cand_by_gene or \
                    len(cb_proteins[r.sseqid]) > len(cb_proteins[cand_by_gene[gene]]):
                cand_by_gene[gene] = r.sseqid
    candidates = list(cand_by_gene.values())

    def score(locus, cid):
        r = bit.get((locus, cid))
        return r.bitscore if r else 0.0

    zims = ['zim-1', 'zim-2', 'zim-3']
    best_assign, best_total = None, -1
    for combo in permutations(candidates, 3):
        total = sum(score(z, c) for z, c in zip(zims, combo))
        if total > best_total:
            best_total, best_assign = total, combo
    assignment = dict(zip(zims, best_assign))
    assignment['him-8'] = him8_protein

    # The reciprocal check: BLAST each chosen C. briggsae protein back against all
    # of C. elegans, and see which C. elegans gene it matches best.
    print("Running reverse blastp (assigned C. briggsae proteins vs C. elegans)...")
    write_fasta(f'{WORK}/assigned.fasta',
                [(cid, cb_proteins[cid]) for cid in assignment.values()])
    rev = blastp(f'{WORK}/assigned.fasta', 'ce', f'{WORK}/reverse.tsv')
    rbh = {}
    for cid in assignment.values():
        sub = rev[rev['qseqid'] == cid].head(1)
        rbh[cid] = ce_headers[sub.iloc[0]['sseqid']].get('locus', '') if len(sub) else ''

    rows = []
    for locus in QUERY_LOCI:
        cid = assignment[locus]
        r = bit.get((locus, cid))
        gene = cb_headers[cid].get('gene')
        chrom, start, end, strand, cb_locus, seqname = coords.get(
            gene, ('', 0, 0, '', '', ''))
        rows.append({
            'ce_paralog': locus,
            'ce_protein': queries[locus][0],
            'cb_protein': cid,
            'cb_gene': seqname or gene,
            'cb_locus': cb_locus,
            'pident': round(r.pident, 1) if r else None,
            'qcovs': int(r.qcovs) if r else None,
            'bitscore': round(r.bitscore, 1) if r else None,
            'evalue': f'{r.evalue:.1e}' if r else None,
            'rbh_locus': rbh[cid],
            'assignment_method': ('curated (Cbr-him-8 name annotation)'
                                  if locus == 'him-8' else 'blastp + 1:1 assignment'),
            'cb_chrom': chrom, 'cb_start': start, 'cb_end': end, 'cb_strand': strand,
        })

    df = pd.DataFrame(rows).sort_values('cb_start')
    df.to_csv(P('ortholog_table_blast.csv'), index=False)

    print("\n" + "=" * 70)
    print("BLAST ortholog assignment (sorted by genomic position):")
    for _, r in df.iterrows():
        ev = r['evalue'] if r['evalue'] is not None else 'n/a'
        print(f"  {r['ce_paralog']:6s} -> {r['cb_protein']:12s} ({r['cb_gene']})  "
              f"id={r['pident']}%  E={ev}  bit={r['bitscore']}  "
              f"chr {r['cb_chrom']} @ {r['cb_start']/1e6:.3f} Mb  [RBH={r['rbh_locus']}]")
    print("\nSaved data/processed/ortholog_table_blast.csv")


if __name__ == '__main__':
    main()
