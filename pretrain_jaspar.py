"""Pretraining data for Baseline 3: learn which parts of an ESM-2 embedding carry
DNA-binding preference, using hundreds of unrelated zinc-finger transcription
factors from JASPAR.

We have only four labelled ZIM/HIM-8 proteins, far too few to learn a 1280-number
embedding -> motif mapping directly. The fix is transfer learning: from many other
C2H2 zinc-finger factors (same domain type as our proteins) we learn a small set
of embedding directions that predict a factor's DNA base preferences, and later
train the tiny head in that reduced space instead of the raw 1280 numbers.

The reduction is done with PLS (partial least squares), which finds the directions
in embedding space that co-vary most with motif composition — i.e. exactly the
"binding-relevant" directions we want to keep.

Pipeline (each stage is cached under data/raw/jaspar and data/processed, so the
slow embedding step is only ever done once):
  1. pull C2H2 factors from the JASPAR REST API (matrix + UniProt id per factor)
  2. fetch each factor's protein sequence from UniProt
  3. embed each factor's zinc-finger domain with the SAME frozen ESM-2 650M model
     and domain read-out used for our four proteins (must match, or the learned
     directions would not apply to them)
  4. fit PLS (embedding -> motif base composition) and save the projection

Leakage guard: any factor from a nematode, or named zim/him-8, is dropped so the
pretraining set stays disjoint from our target family.

Run from the project root:  python pretrain_jaspar.py
Writes data/processed/jaspar_projection.npz (used automatically by train_head.py).
"""
import json
from src.paths import P
import os
import time

import numpy as np
import requests

from src.embeddings import embed_sequences, zinc_finger_domain, DEFAULT_MODEL

RAW_DIR = 'data/raw/jaspar'
META_FILE = f'{RAW_DIR}/c2h2_meta.json'
SEQ_FILE = f'{RAW_DIR}/sequences.json'
EMB_FILE = P('jaspar_embeddings.npz')
PROJ_FILE = P('jaspar_projection.npz')

API = 'https://jaspar.genereg.net/api/v1/matrix/'
UNIPROT = 'https://rest.uniprot.org/uniprotkb/{}.fasta'
MAX_FACTORS = 300           # cap the corpus so the CPU embedding step stays bounded
N_COMPONENTS = 12           # PLS directions to keep (the reduced feature space)
LEAKAGE_TERMS = ('zim', 'him-8', 'caenorhabditis', 'elegans', 'briggsae', 'nematode')


def fetch_metadata():
    """Collect C2H2 zinc-finger factors from JASPAR: matrix id, UniProt id, PFM.

    Cached to disk; only the first run hits the network.
    """
    if os.path.exists(META_FILE):
        return json.load(open(META_FILE))
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Listing C2H2 zinc-finger factors from JASPAR CORE...")
    ids, url, params = [], API, {'collection': 'CORE',
                                 'tf_class': 'C2H2 zinc finger factors',
                                 'page_size': 100}
    while url:
        r = requests.get(url, params=params, timeout=30).json()
        ids += [m['matrix_id'] for m in r['results']]
        url, params = r.get('next'), None
    print(f"  {len(ids)} matrices found; fetching details (UniProt id + PFM)...")

    records = []
    for i, mid in enumerate(ids):
        try:
            d = requests.get(f'{API}{mid}/', timeout=30).json()
        except Exception:
            continue
        uni = (d.get('uniprot_ids') or [None])[0]
        name = (d.get('name') or '').lower()
        species = ' '.join(s.get('name', '') for s in d.get('species', [])).lower()
        if not uni or not d.get('pfm'):
            continue
        if any(t in name or t in species for t in LEAKAGE_TERMS):
            continue
        records.append({'matrix_id': mid, 'name': d.get('name'),
                        'uniprot': uni, 'pfm': d['pfm'], 'species': species})
        if (i + 1) % 100 == 0:
            print(f"    ...{i + 1}/{len(ids)}")
        time.sleep(0.02)
    json.dump(records, open(META_FILE, 'w'))
    print(f"  kept {len(records)} factors with UniProt id + PFM")
    return records


def fetch_sequences(records):
    """Download each factor's protein sequence from UniProt (cached)."""
    seqs = json.load(open(SEQ_FILE)) if os.path.exists(SEQ_FILE) else {}
    todo = [r['uniprot'] for r in records if r['uniprot'] not in seqs]
    todo = list(dict.fromkeys(todo))
    print(f"Fetching {len(todo)} protein sequences from UniProt...")
    for i, uni in enumerate(todo):
        try:
            txt = requests.get(UNIPROT.format(uni), timeout=30).text
            seq = ''.join(l.strip() for l in txt.splitlines() if not l.startswith('>'))
            if seq:
                seqs[uni] = seq
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"    ...{i + 1}/{len(todo)}")
            json.dump(seqs, open(SEQ_FILE, 'w'))
    json.dump(seqs, open(SEQ_FILE, 'w'))
    return seqs


def embed_factors(records, seqs):
    """Embed each factor's zinc-finger domain with the frozen ESM-2 650M model.

    Uses the identical domain read-out as our four target proteins, so the
    resulting vectors live in the same space. Cached; resumable.
    """
    cache = dict(np.load(EMB_FILE)) if os.path.exists(EMB_FILE) else {}
    pending = []
    for r in records:
        uni = r['uniprot']
        if uni in cache or uni not in seqs:
            continue
        seq = seqs[uni]
        region = zinc_finger_domain(seq)
        if region is None or len(seq) < 40:
            continue
        pending.append((uni, seq, region))

    pending = pending[:max(0, MAX_FACTORS - len(cache))]
    print(f"Embedding {len(pending)} factor domains with {DEFAULT_MODEL} "
          f"(cached so far: {len(cache)})...")
    BATCH = 16
    for i in range(0, len(pending), BATCH):
        chunk = pending[i:i + BATCH]
        vecs = embed_sequences([s for _, s, _ in chunk],
                               regions=[reg for _, _, reg in chunk])
        for (uni, _, _), v in zip(chunk, vecs):
            cache[uni] = v
        np.savez(EMB_FILE, **cache)
        print(f"    embedded {min(i + BATCH, len(pending))}/{len(pending)}")
    return dict(np.load(EMB_FILE))


def motif_features(pfm):
    """Summarise a motif as a fixed-length base-preference vector.

    Combines the average single-letter frequencies (4 numbers) with the average
    neighbouring-letter-pair frequencies (16 numbers) across the motif, giving a
    20-number fingerprint of the DNA sequence the factor prefers.
    """
    M = np.array([pfm[b] for b in 'ACGT'], dtype=float)  # (4, L)
    M = M / M.sum(axis=0, keepdims=True)
    mono = M.mean(axis=1)                                  # (4,)
    di = np.zeros((4, 4))
    for j in range(M.shape[1] - 1):
        di += np.outer(M[:, j], M[:, j + 1])
    di = (di / max(1, M.shape[1] - 1)).ravel()            # (16,)
    return np.concatenate([mono, di])


def main():
    from sklearn.cross_decomposition import PLSRegression

    records = fetch_metadata()
    seqs = fetch_sequences(records)
    emb = embed_factors(records, seqs)

    X, Y = [], []
    for r in records:
        if r['uniprot'] in emb:
            X.append(emb[r['uniprot']])
            Y.append(motif_features(r['pfm']))
    X, Y = np.vstack(X), np.vstack(Y)

    # Drop any factor whose embedding or motif features came out non-finite
    # (e.g. a degenerate domain slice) — PLS cannot handle NaNs.
    good = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if not good.all():
        print(f"  dropping {int((~good).sum())} factors with non-finite features")
    X, Y = X[good], Y[good]
    print(f"\nPretraining matrix: {X.shape[0]} factors x {X.shape[1]} embedding dims"
          f" -> {Y.shape[1]} motif features")

    mean = X.mean(axis=0)
    Xc = X - mean
    r = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])
    pls = PLSRegression(n_components=r, scale=False)
    pls.fit(Xc, Y - Y.mean(axis=0))
    projection = pls.x_rotations_               # (embedding_dim, r)

    # How much motif variance the projection explains, as a sanity signal.
    Yci = Y - Y.mean(axis=0)
    pred = pls.predict(Xc)              # predictions of the centered motif features
    ss = 1 - np.sum((Yci - pred) ** 2) / np.sum(Yci ** 2)
    print(f"PLS kept {r} components; in-sample motif-feature R^2 = {ss:.3f}")

    np.savez(PROJ_FILE, projection=projection, mean=mean, n_components=r)
    meta = {'model': DEFAULT_MODEL, 'n_factors': int(X.shape[0]),
            'n_components': int(r), 'motif_features': 'mono(4)+dinucleotide(16)',
            'source': 'JASPAR CORE C2H2 zinc finger factors',
            'insample_r2': float(ss)}
    json.dump(meta, open(P('jaspar_projection_meta.json'), 'w'), indent=2)
    print(f"\nSaved {PROJ_FILE} (projection {projection.shape}) + meta. "
          f"Now re-run train_head.py for the pretrained setting.")


if __name__ == '__main__':
    main()
