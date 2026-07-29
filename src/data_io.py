"""Functions for loading the raw data files.

Two file types are used throughout the project:
  - FASTA files, which hold DNA or protein sequences
  - GFF3 files, which hold gene annotations (where each gene sits on a chromosome)

Keeping the loading code in one place means every part of the project reads the
data the same way.
"""
import gzip

import pandas as pd

# A GFF3 file has no header row, so we name its nine columns ourselves.
GFF_COLUMNS = ['seqid', 'source', 'type', 'start', 'end',
               'score', 'strand', 'phase', 'attributes']


def read_fasta(file_path):
    """Read a FASTA sequence file into a dictionary of {name: sequence}.

    A FASTA file lists sequences, each starting with a ">" header line (the
    name) followed by the sequence itself. We keep the first word of the header
    as the name. Compressed files (ending in .gz) are handled automatically.
    """
    open_fn = gzip.open if str(file_path).endswith('.gz') else open
    sequences = {}
    with open_fn(file_path, 'rt') as f:
        seq_id = ""
        seq_list = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # A new header means the previous sequence is finished; save it.
                if seq_id:
                    sequences[seq_id] = "".join(seq_list)
                seq_id = line[1:].split()[0]
                seq_list = []
            else:
                # Sequences can span many lines, so we collect them and join later.
                seq_list.append(line)
        # Don't forget the final sequence in the file.
        if seq_id:
            sequences[seq_id] = "".join(seq_list)
    return sequences


def read_fasta_headers(file_path):
    """Read the full description line of each sequence in a FASTA file.

    read_fasta keeps only the sequence name (the first word). But the rest of
    the header holds useful labels, such as the gene name (for example
    "locus=zim-1"). This function returns the whole description so we can search
    those labels. It returns a dictionary of {name: full description}.
    """
    open_fn = gzip.open if str(file_path).endswith('.gz') else open
    headers = {}
    with open_fn(file_path, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                description = line[1:].strip()
                headers[description.split()[0]] = description
    return headers


def read_gff(file_path):
    """Read a GFF3 gene-annotation file into a table (a pandas DataFrame).

    Lines starting with "#" are comments and are skipped. Compressed files are
    handled automatically.
    """
    return pd.read_csv(
        file_path,
        sep='\t',
        comment='#',
        header=None,
        names=GFF_COLUMNS,
    )


def extract_gene_id(attributes_str):
    """Pull the gene's ID out of the free-text "attributes" field of a GFF3 row.

    That field packs several labels together, like "ID=Gene:WBGene...;Name=...".
    We return the first ID-like value we find, or nothing if there isn't one.
    """
    for part in attributes_str.split(';'):
        part = part.strip()
        if part.startswith(('ID=', 'Name=', 'gene_id=')):
            return part.split('=')[1].strip('"')
    return None
