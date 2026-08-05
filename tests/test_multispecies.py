"""Tests for the multi-species pipeline, run on synthetic data.

The genomes are large and not in git, so these tests build a small artificial world
instead: several species, four paralogs each, and motifs that vary in a way we
control exactly. That lets us check the parts that are easy to get quietly wrong:

  * the metric behaves (a perfect prediction scores 1, a blind guess scores 0)
  * the canonical frame survives a round trip
  * the head can learn a signal that IS there — and, just as important, does NOT
    beat the group-consensus baseline when the embeddings are pure noise. A model
    that "wins" on noise is measuring a leak, not a signal.
  * leave-one-species-out really holds the species out
  * the annotation gate excludes incomplete annotations

Run:  python tests/test_multispecies.py     (or: python -m pytest tests/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motifs import (canonical_to_pwm, motif_to_pwm, pwm_to_canonical,
                        CANON_VARIABLE)
from src.pcmodel import (Example, consensus_canonical, evaluate_split, gold_test,
                         group_consensus_canonical, leave_one_species_out,
                         prepare_features, score_canonical, score_native, summarize,
                         uniform_canonical)
from src.species import normalize_chrom

PARALOGS = ['HIM-8', 'ZIM-1', 'ZIM-2', 'ZIM-3']
SPACER_LETTERS = {'HIM-8': 'AC', 'ZIM-1': 'CA', 'ZIM-2': 'GT', 'ZIM-3': 'TA'}
CLASS_LETTERS = 'ACG'          # the within-group variation the model must learn
CLASS_TAILS = {0: 'ACG', 1: 'CGA', 2: 'GAC'}
SPACER = 2


def synthetic_motif(paralog, cls):
    """TTGG + paralog-specific spacer + TG + species-class tail + one fixed letter.

    The split is deliberate. The spacer letters depend only on the paralog, so the
    group-consensus baseline gets them for free — as it would on real data, where
    labels are joined to proteins by paralog. The first three tail letters depend
    only on the species class, so the ONLY way to beat group-consensus is to read
    that class out of the embedding, which is exactly the ability being tested.
    """
    return 'TTGG' + SPACER_LETTERS[paralog] + 'TG' + CLASS_TAILS[cls] + 'T'


def synthetic_examples(n_species=6, noise=0.0, seed=0, signal=True):
    """Build a small labelled world.

    Species i belongs to class i % 3, so holding out any one species still leaves its
    class represented in the training set — otherwise the class could never be
    learned and the test would be measuring nothing.

    With `signal=False` the embedding carries no class information, which is the
    negative control.
    """
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(n_species):
        cls = i % len(CLASS_LETTERS)
        for j, paralog in enumerate(PARALOGS):
            vec = np.zeros(4 + len(CLASS_LETTERS))
            vec[j] = 1.0                                  # which paralog
            if signal:
                vec[4 + cls] = 1.0                        # which species class
            vec = vec + rng.normal(0, noise, vec.shape)
            examples.append(Example(
                key=f'S{i}_{paralog.lower()}', species=f'species_{i}',
                paralog=paralog, vector=vec,
                motif=synthetic_motif(paralog, cls), spacer=SPACER, weight=1.0))
    return examples


# ---------------------------------------------------------------------------

def test_metric_endpoints():
    """A perfect prediction scores 1; a uniform guess scores 0."""
    e = synthetic_examples(n_species=1)[0]
    perfect = score_canonical(e.canon, e.canon, e.mask)
    assert abs(perfect['skill'] - 1.0) < 1e-9, perfect
    blind = score_canonical(uniform_canonical(), e.canon, e.mask)
    assert abs(blind['skill']) < 1e-9, blind

    native_perfect = score_native(e.pwm, e.pwm, e.spacer)
    assert abs(native_perfect['skill'] - 1.0) < 1e-9, native_perfect
    print("  metric endpoints (1.0 perfect / 0.0 blind)            ok")


def test_canonical_round_trip():
    """Mapping a motif into the shared 9 slots and back preserves it."""
    for paralog in PARALOGS:
        motif = synthetic_motif(paralog, 0)
        pwm = motif_to_pwm(motif)
        canon, mask = pwm_to_canonical(pwm, SPACER)
        assert mask.shape == (CANON_VARIABLE,)
        rebuilt = canonical_to_pwm(canon, SPACER, len(motif), anchor_motif=motif)
        # Every graded position must come back unchanged.
        assert np.allclose(rebuilt, pwm, atol=1e-9), paralog
    print("  canonical frame round trip                            ok")


def test_only_used_slots_are_graded():
    """A protein is graded on the slots it uses, not on slots it does not have."""
    e = synthetic_examples(n_species=1)[0]
    assert e.mask.sum() == 6, e.mask          # 2 spacer + 4 tail slots for this motif
    res = score_canonical(uniform_canonical(), e.canon, e.mask)
    assert res['n_slots'] == 6, res
    print("  grading restricted to the protein's own slots         ok")


def test_group_consensus_ignores_embedding():
    """The group-consensus baseline must depend only on the paralog group."""
    examples = synthetic_examples()
    idx = list(range(len(examples)))
    a = group_consensus_canonical(examples, idx, 'ZIM-1')
    # Perturbing every embedding must not change this baseline at all.
    for e in examples:
        e.vector = e.vector + 10.0
    b = group_consensus_canonical(examples, idx, 'ZIM-1')
    assert np.allclose(a, b), "group-consensus must not read the embedding"
    print("  group-consensus baseline ignores embeddings           ok")


def test_head_learns_real_signal():
    """When the embedding really does predict the motif, the head must beat the bar.

    The spacer letters are fixed per paralog (group-consensus gets them free), so the
    head can only win by learning the species-class letter from the embedding.
    """
    examples = synthetic_examples(noise=0.05, signal=True)
    rows = leave_one_species_out(examples, l2=0.03, epochs=600)
    table = summarize(rows).set_index('method')
    head = table.loc['learned-head', 'skill']
    group = table.loc['group-consensus', 'skill']
    floor = table.loc['floor (uniform)', 'skill']
    assert abs(floor) < 1e-6, f"floor should be 0, got {floor}"
    assert head > group + 0.05, f"head {head:.3f} should beat group {group:.3f}"
    print(f"  head beats group-consensus on real signal "
          f"({head:+.3f} vs {group:+.3f})   ok")


def test_head_does_not_win_on_noise():
    """The negative control: with no signal, the head must NOT beat the bar.

    This is the guard that makes the positive result meaningful. If a held-out
    species could leak into its own prediction — through the centering mean, the
    feature scale, or the group consensus — the head would beat group-consensus even
    here, and the whole evaluation would be worthless.
    """
    examples = synthetic_examples(noise=1.0, signal=False, seed=7)
    rows = leave_one_species_out(examples, l2=1.0, epochs=600)
    table = summarize(rows).set_index('method')
    head = table.loc['learned-head', 'skill']
    group = table.loc['group-consensus', 'skill']
    assert head <= group + 0.02, (
        f"head {head:.3f} beat group {group:.3f} on pure noise — this indicates "
        f"leakage from the held-out species into its own prediction")
    print(f"  head does not beat the bar on noise "
          f"({head:+.3f} vs {group:+.3f})         ok")


def test_loso_holds_the_species_out():
    """Every held-out row must belong to the species being held out."""
    examples = synthetic_examples()
    rows = leave_one_species_out(examples, l2=1.0, epochs=100)
    species = sorted({e.species for e in examples})
    for r in rows:
        assert r['species'] == r['held_out_species'], r
    assert sorted({r['held_out_species'] for r in rows}) == species
    print(f"  leave-one-species-out holds each of {len(species)} species out    ok")


def test_features_use_training_split_only():
    """Centering and scaling must be fit on the training split alone."""
    examples = synthetic_examples()
    all_idx = list(range(len(examples)))
    subset = all_idx[:8]
    f_all = prepare_features(examples, all_idx)
    f_sub = prepare_features(examples, subset)
    assert not np.allclose(f_all, f_sub), (
        "features did not change when the training split changed, so the centering "
        "mean is not being fit on the training split")
    print("  centering/scaling fit on the training split only      ok")


def test_gold_examples_are_not_trained_on():
    """The gold test must train on silver labels only."""
    silver = synthetic_examples(n_species=4)
    gold = [Example(key=f'Cel_{p.lower()}', species='c_elegans', paralog=p,
                    vector=np.eye(7)[i], motif=synthetic_motif(p, 0), spacer=SPACER,
                    weight=1.0, label_type='gold') for i, p in enumerate(PARALOGS)]
    rows = gold_test(silver, gold, l2=1.0, epochs=100)
    assert {r['species'] for r in rows} == {'c_elegans'}
    assert {r['label_type'] for r in rows} == {'gold'}
    assert len(rows) == len(gold) * 5          # five methods scored per gold protein
    print("  gold test trains on silver only, tests on gold        ok")


def test_zero_weight_labels_are_excluded():
    """A label with weight 0 (a LOW-confidence discovery) must not train the model."""
    examples = synthetic_examples(n_species=4)
    idx = list(range(len(examples)))
    baseline = consensus_canonical(examples, idx)
    for e in examples:
        if e.paralog == 'ZIM-1':
            e.weight = 0.0
    dropped = consensus_canonical(examples, idx)
    assert not np.allclose(baseline, dropped), "zero-weight labels still influenced the fit"
    print("  zero-weight labels excluded from training             ok")


def test_chromosome_normalization():
    """Assembly sequence names map onto the six real chromosome names."""
    cases = {'I': 'I', 'chrI': 'I', 'CHRX': 'X', 'chr_V': 'V', 'chromosome_II': 'II',
             'MtDNA': None, 'scaffold_123': None, 'X': 'X'}
    for name, expected in cases.items():
        got = normalize_chrom(name)
        assert got == expected, f"{name!r} -> {got!r}, expected {expected!r}"
    # An explicit map handles accession-style names.
    assert normalize_chrom('CM008945.1', {'CM008945.1': 'IV'}) == 'IV'
    print("  chromosome name normalization                         ok")


def test_annotation_gate_rejects_incomplete(tmp_root='/tmp/_pc_gate_test'):
    """A species whose annotation is declared incomplete must be excluded.

    Also checks the independent verification: a species declared complete but
    carrying a stub proteome is still rejected, and the reason is recorded.
    """
    import shutil

    from src.species import Species, assess

    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)

    genome = os.path.join(tmp_root, 'g.fa')
    with open(genome, 'w') as f:
        for c in ['I', 'II', 'III', 'IV', 'V', 'X']:
            f.write(f'>{c}\n' + 'ACGT' * 1_500_000 + '\n')
    proteins = os.path.join(tmp_root, 'p.fa')
    with open(proteins, 'w') as f:
        for i in range(40):                      # a stub proteome
            f.write(f'>prot{i}\nMKV\n')
    gff = os.path.join(tmp_root, 'a.gff3')
    with open(gff, 'w') as f:
        for i in range(40):
            f.write(f'I\ttest\tgene\t{i}\t{i + 10}\t.\t+\t.\tID=g{i}\n')

    declared_incomplete = Species(key='sp_bad', name='Bad', short='Bad', genome=genome,
                                  proteins=proteins, annotation=gff,
                                  declared_complete=False, notes='listed as incomplete')
    rep = assess(declared_incomplete)
    assert not rep['usable_for_labels'] and not rep['usable_for_proteins']
    assert 'declared incomplete' in rep['reasons']

    # Declared complete, but the files say otherwise.
    declared_complete = Species(key='sp_stub', name='Stub', short='Stu', genome=genome,
                                proteins=proteins, annotation=gff,
                                declared_complete=True)
    rep2 = assess(declared_complete)
    assert not rep2['usable_for_proteins'], "a stub proteome must not pass the gate"
    assert 'proteins' in rep2['reasons']

    missing = Species(key='sp_missing', name='M', short='M', genome=genome,
                      proteins='', annotation='', declared_complete=True)
    rep3 = assess(missing)
    assert not rep3['usable_for_proteins'] and 'missing files' in rep3['reasons']

    shutil.rmtree(tmp_root, ignore_errors=True)
    print("  annotation gate rejects incomplete annotations        ok")


def test_evaluate_split_scores_every_method():
    """Each held-out protein is scored by all five methods, with native-frame skill."""
    examples = synthetic_examples(n_species=4)
    train = [i for i, e in enumerate(examples) if e.species != 'species_0']
    test = [i for i, e in enumerate(examples) if e.species == 'species_0']
    rows = evaluate_split(examples, train, test, l2=1.0, epochs=100)
    methods = {r['method'] for r in rows}
    assert methods == {'floor (uniform)', 'global-consensus', 'group-consensus',
                       'nearest-neighbour', 'learned-head'}, methods
    assert all('native_skill' in r for r in rows)
    assert all(-2.0 <= r['skill'] <= 1.0 for r in rows)
    print("  evaluate_split scores all five methods                ok")


def test_discovery_recovers_a_planted_motif():
    """The label-generating path, end to end, on a genome whose answer we know.

    A synthetic genome gets a tandem array of a known motif planted near one end of
    each chromosome — which is what a real pairing center is. Discovery has to find
    the array with no hint of where it is or what it says, and the synteny join has
    to hand each recovered motif to the right paralog. If this breaks, every silver
    label is wrong and nothing downstream means anything.
    """
    from src.silver import CE_CHROM_PARALOG, build_labels, discover_species, species_seeds

    planted = {'III': 'TTGGTCTGCTAATTAT',      # spacer 2, ZIM-1's chromosome
               'IV': 'TTGGGTCATGACCTAG',       # spacer 4, ZIM-3's chromosome
               'X': 'TTGGTAGTGGTTCCGC'}        # spacer 3, HIM-8's chromosome
    rng = np.random.default_rng(0)
    genome = {}
    for chrom, motif in planted.items():
        # At least 1 Mb, because the enrichment background only counts sequences
        # that size — the same threshold a real chromosome comfortably clears.
        seq = list(''.join(rng.choice(list('ACGT'), 1_050_000)))
        for i in range(60):                    # a tandem array: 60 copies, 220 bp apart
            p = 20_000 + i * 220
            seq[p:p + len(motif)] = list(motif)
        genome[chrom] = ''.join(seq)

    seeds = species_seeds(genome, terminal_size=300_000, min_count=5)
    records = discover_species(genome, seeds)
    found = {r['chromosome']: r for r in records}
    assert set(found) == set(planted), f"discovered {sorted(found)}, expected {sorted(planted)}"
    for chrom, motif in planted.items():
        assert found[chrom]['motif'] == motif, (
            f"chr{chrom}: recovered {found[chrom]['motif']}, planted {motif}")
        assert found[chrom]['confidence'] == 'HIGH', found[chrom]

    orthologs = {p: {'protein_id': f'PROT_{p}', 'identity': 60.0, 'score': 500.0}
                 for p in PARALOGS}
    labels = build_labels('test_sp', 'Test species', 'Tsp', orthologs, records)
    assert len(labels) == len(planted)
    for lab in labels:
        assert lab['used_for_training'] and lab['weight'] == 1.0, lab
        assert lab['paralog'] == CE_CHROM_PARALOG[lab['chromosome']], lab
    print("  discovery recovers planted motifs and joins by synteny ok")


def test_orthologs_assigned_one_to_one():
    """Each paralog gets its own protein; no protein is handed to two paralogs.

    These four paralogs are recent duplicates and are more similar to each other than
    to anything else, so a naive per-query best hit would happily assign the same
    protein twice. The assignment has to be globally greedy and exclusive.
    """
    from src.silver import find_orthologs_in

    rng = np.random.default_rng(3)
    aa = list('ACDEFGHIKLMNPQRSTVWY')

    def mutate(seq, frac):
        s = list(seq)
        for i in rng.choice(len(s), int(len(s) * frac), replace=False):
            s[i] = rng.choice(aa)
        return ''.join(s)

    base = ''.join(rng.choice(aa, 260))
    queries, proteome = {}, {}
    for i, paralog in enumerate(PARALOGS):
        q = mutate(base, 0.25)                       # the four paralogs: similar
        queries[paralog] = q
        proteome[f'ORTH_{paralog}'] = mutate(q, 0.20)   # its ortholog: closest to it
    for d in range(6):
        proteome[f'DECOY{d}'] = ''.join(rng.choice(aa, 260))

    assigned = find_orthologs_in(proteome, queries)
    ids = [a['protein_id'] for a in assigned.values()]
    assert len(ids) == len(set(ids)), f"a protein was assigned twice: {ids}"
    assert set(assigned) == set(PARALOGS), f"unassigned paralogs: {set(PARALOGS) - set(assigned)}"
    for paralog, a in assigned.items():
        assert a['protein_id'] == f'ORTH_{paralog}', (
            f"{paralog} matched {a['protein_id']}, expected ORTH_{paralog}")
    print("  orthologs assigned one-to-one, no protein reused       ok")


def test_species_prefixes_stay_unique():
    """Two assemblies of one species must not end up sharing a protein prefix.

    'c_briggsae_af16' and 'c_briggsae_qx1410' both reduce to 'Cbr'. Since the prefix
    plus the paralog is how a protein is looked up, a collision would make one
    species' embeddings silently overwrite the other's.
    """
    import csv
    import tempfile

    from prepare_species import _short
    from src.species import load_manifest

    used = set()
    shorts = []
    for key in ['c_briggsae_af16', 'c_briggsae_qx1410', 'c_remanei', 'c_elegans']:
        s = _short(key, used)
        used.add(s)
        shorts.append(s)
    assert len(shorts) == len(set(shorts)), f"colliding prefixes: {shorts}"

    # A hand-edited manifest with a duplicate prefix must fail loudly, not silently.
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['key', 'name', 'short', 'genome', 'proteins', 'annotation',
                    'annotation_complete', 'notes', 'chrom_map'])
        w.writerow(['sp_a', 'A', 'Cbr', 'a.fa', 'a.pep', 'a.gff3', 'yes', '', ''])
        w.writerow(['sp_b', 'B', 'Cbr', 'b.fa', 'b.pep', 'b.gff3', 'yes', '', ''])
        path = fh.name
    try:
        raised = False
        try:
            load_manifest(path)
        except ValueError as exc:
            raised = 'Duplicate short' in str(exc)
        assert raised, "a manifest with two species sharing a prefix must be rejected"
    finally:
        os.unlink(path)
    print("  species prefixes stay unique                          ok")


def test_shipped_manifest_is_valid():
    """The manifest committed to the repo must load and have unique keys/prefixes."""
    from src.species import load_manifest

    species = load_manifest('data/species_manifest.csv')
    assert species, "shipped manifest is empty"
    assert len({s.short for s in species}) == len(species)
    assert any('elegans' in s.key for s in species), (
        "C. elegans must be in the manifest: it supplies the ortholog queries and "
        "the gold test set")
    print(f"  shipped manifest loads ({len(species)} species)           ok")


TESTS = [test_metric_endpoints, test_canonical_round_trip,
         test_only_used_slots_are_graded, test_group_consensus_ignores_embedding,
         test_head_learns_real_signal, test_head_does_not_win_on_noise,
         test_loso_holds_the_species_out, test_features_use_training_split_only,
         test_gold_examples_are_not_trained_on, test_zero_weight_labels_are_excluded,
         test_chromosome_normalization, test_annotation_gate_rejects_incomplete,
         test_evaluate_split_scores_every_method,
         test_discovery_recovers_a_planted_motif, test_orthologs_assigned_one_to_one,
         test_species_prefixes_stay_unique, test_shipped_manifest_is_valid]


if __name__ == '__main__':
    print("Multi-species pipeline tests (synthetic data)\n")
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"  FAILED {t.__name__}: {exc}")
        except Exception as exc:                       # noqa: BLE001 - report and continue
            failed += 1
            print(f"  ERROR  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
