"""Turning protein sequences into numbers a model can learn from (ESM-2 embeddings).

ESM-2 is a large "protein language model": trained on millions of proteins, it can
read a protein sequence and summarize it as a list of numbers (a vector) that
captures its structural and functional features. We use it purely as a fixed
feature extractor: we run each protein through it once, save the resulting vector,
and never adjust the model itself. That is deliberate, because we only have four
training proteins, far too few to retrain something this large.

The model name is kept in one place (DEFAULT_MODEL) so it is easy to switch
between a small, fast version and the large, more accurate version.

By default we summarize a whole protein by averaging the per-position vectors into
one vector. We can also summarize just a chosen stretch of the protein (see the
regions option below), which is useful for focusing on the DNA-binding region.
"""
import re

import numpy as np

# The protein language model to use. The small 'facebook/esm2_t12_35M_UR50D' is
# faster and fine for wiring things up; this larger one is more accurate.
DEFAULT_MODEL = 'facebook/esm2_t33_650M_UR50D'

# A rough sequence pattern for a zinc finger: two Cysteines (C) and two
# Histidines (H) spaced a certain distance apart. These proteins have two of
# these fingers in a row, and the DNA-binding region is those two fingers plus
# the stretch after them, running to the end of the protein.
_ZF_PATTERN = re.compile(r'C.{2,5}C.{7,20}H.{2,6}H')

_MODEL_CACHE = {}


def zinc_finger_domain(seq, margin=5):
    """Locate the DNA-binding region of a protein: the start and end positions.

    It runs from just before the first zinc finger to the end of the protein.
    Everything before that (a long, floppy region that carries little
    binding information) is left out. Returns nothing if no zinc finger is found.
    """
    matches = list(_ZF_PATTERN.finditer(seq))
    if not matches:
        return None
    return (max(0, matches[0].start() - margin), len(seq))


def _load(model_name):
    """Download the model the first time it's needed and keep it in memory.

    The model is put in read-only mode ("frozen") so it is never changed.
    """
    if model_name not in _MODEL_CACHE:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        for p in model.parameters():       # freeze: never update these weights
            p.requires_grad = False
        _MODEL_CACHE[model_name] = (tokenizer, model, torch)
    return _MODEL_CACHE[model_name]


def embed_sequences(seqs, model_name=DEFAULT_MODEL, pooling='mean', regions=None):
    """Turn each protein sequence into one summary vector of numbers.

    The model produces a vector for every position in the protein; by default we
    average those into a single vector per protein. If 'regions' is given (a
    start and end position for each protein), we instead average only over that
    stretch, for example the DNA-binding region. Importantly, the model still
    reads the whole protein for context either way; we simply choose which part
    to summarize.
    """
    tokenizer, model, torch = _load(model_name)
    vectors = []
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            enc = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
            # One vector per position in the protein.
            out = model(**enc).last_hidden_state[0]
            mask = enc['attention_mask'][0].bool()
            # Drop the two internal marker positions the model adds at each end.
            residues = out[mask][1:-1]
            region = regions[i] if regions is not None else None
            if region is not None:
                residues = residues[region[0]:region[1]]  # keep only the chosen stretch
            if pooling == 'mean':
                vec = residues.mean(dim=0)                 # average into one vector
            elif pooling == 'cls':
                vec = out[0]
            else:
                raise ValueError(f"unknown pooling: {pooling}")
            vectors.append(vec.numpy())
    return np.vstack(vectors)


def embed_positions(seqs, positions, model_name=DEFAULT_MODEL):
    """Summarize each protein using only a chosen SET of residue positions.

    Unlike embed_sequences (which averages over a contiguous stretch), this
    averages the per-residue vectors at an arbitrary list of positions — for
    example the recognition-helix residues of the two zinc fingers, which need not
    be contiguous. The model still reads the whole protein for context; we simply
    pool over the selected positions. `positions[i]` is the list of 0-based
    residue indices to keep for sequence i.
    """
    tokenizer, model, torch = _load(model_name)
    vectors = []
    with torch.no_grad():
        for seq, pos in zip(seqs, positions):
            enc = tokenizer(seq, return_tensors='pt', truncation=True, max_length=1024)
            out = model(**enc).last_hidden_state[0]
            mask = enc['attention_mask'][0].bool()
            residues = out[mask][1:-1]                 # one vector per residue
            idx = [p for p in pos if 0 <= p < residues.shape[0]]
            if not idx:
                raise ValueError("no valid recognition positions for a sequence")
            vectors.append(residues[idx].mean(dim=0).numpy())
    return np.vstack(vectors)


def embedding_dim(model_name=DEFAULT_MODEL):
    """How many numbers are in each protein's summary vector for a given model."""
    _, model, _ = _load(model_name)
    return model.config.hidden_size
