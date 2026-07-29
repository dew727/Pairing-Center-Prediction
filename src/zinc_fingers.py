"""Locating the DNA-recognition parts of the two atypical zinc fingers.

The ZIM/HIM-8 proteins bind DNA through two C2H2 zinc fingers, and the letters a
finger recognises are set by a short stretch of residues in its recognition helix
— the segment between the finger's second cysteine and its first histidine. The
gene-conversion analysis (Liu 2009) shows that everything N-terminal to these
fingers has been homogenised between paralogs and carries no paralog-specific
information, so for telling the proteins apart we want to look only at the
recognition helices, not the whole protein.

These are *atypical* C2H2 fingers (non-canonical cysteine/histidine spacing —
which is why general structure predictors mis-handle this family), so a permissive
pattern is used rather than the strict C2H2 consensus.
"""
import re

# Permissive C2H2 finger: two cysteines, a recognition stretch, two histidines.
# The captured group is the recognition helix (between the 2nd Cys and 1st His);
# non-greedy so it stays tight to the actual helix rather than running downstream.
_FINGER = re.compile(r'C.{2,5}C(.{7,20}?)H.{2,6}H')


def find_fingers(seq):
    """Return each zinc finger's recognition-helix span as (start, end) indices.

    Indices are 0-based into the protein sequence and cover the residues between
    the finger's second cysteine and its first histidine.
    """
    return [(m.start(1), m.end(1)) for m in _FINGER.finditer(seq)]


def recognition_positions(seq, n_fingers=2):
    """List the recognition-helix residue positions for the first `n_fingers`.

    Returns one list of residue indices per finger (empty if the finger is not
    found), so callers can read out embeddings at exactly those positions.
    """
    spans = find_fingers(seq)[:n_fingers]
    return [list(range(a, b)) for a, b in spans]


def recognition_residues(seq, n_fingers=2):
    """The recognition-helix amino-acid sequence of each finger (for inspection)."""
    return [seq[a:b] for a, b in find_fingers(seq)[:n_fingers]]


# End of a finger match (past its second histidine), used to locate the CTD.
_FINGER_FULL = re.compile(r'C.{2,5}C.{7,20}?H.{2,6}H')


def ctd_region(seq, n_fingers=2):
    """Locate the carboxyl-terminal domain (CTD): from just after the last zinc
    finger to the end of the protein.

    Li et al. 2024 identify the CTD — not the conserved base-contacting residues —
    as the region that sets ZF1-2 conformation and thereby the paralog's motif
    specificity. Returns (start, end) or None if fewer than n_fingers are found.
    """
    ends = [m.end() for m in _FINGER_FULL.finditer(seq)]
    if len(ends) < n_fingers:
        return None
    return (ends[n_fingers - 1], len(seq))
