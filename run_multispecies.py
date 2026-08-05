"""Run the whole multi-species pipeline with one command.

The four stages have to run in order and each depends on the last, so this driver
runs them, checks the preconditions each one needs, and skips work that is already
done. Re-running it after an interruption picks up where it left off.

    python run_multispecies.py                 # run everything that isn't done yet
    python run_multispecies.py --scan          # also (re)draft the species manifest first
    python run_multispecies.py --force         # redo every stage from scratch
    python run_multispecies.py --from labels   # restart at a given stage
    python run_multispecies.py --domain        # train on whole-domain features instead

Stages:
    manifest   draft data/species_manifest.csv from what is in data/raw/   (--scan only)
    gate       apply the annotation-completeness gate     -> species_qc.csv
    labels     orthologs + per-species motif discovery    -> silver_labels.csv
    embed      ESM-2 features for every species' proteins -> multispecies_embeddings_*
    train      leave-one-species-out + gold test          -> multispecies_summary.csv

`labels` is the slow one: it counts k-mers across every genome, a few minutes per
species. Its per-species results are cached, so a later re-run is fast.
"""
import os
import subprocess
import sys
import time

from src.paths import P

STAGES = ['gate', 'labels', 'embed', 'train']

# What each stage runs, and the file that proves it already finished.
STEPS = {
    'gate': (['prepare_species.py'], lambda: P('species_qc.csv')),
    'labels': (['build_multispecies_labels.py'], lambda: P('silver_labels.csv')),
    'embed': (['embed_multispecies.py'],
              lambda: P('multispecies_embeddings_recognition.npz')),
    'train': (['train_multispecies.py'], lambda: P('multispecies_summary.csv')),
}


def _run(script, extra_args=()):
    """Run one stage as a subprocess, streaming its output, and stop on failure."""
    cmd = [sys.executable, script, *extra_args]
    print(f"\n{'=' * 74}\n$ {' '.join(cmd)}\n{'=' * 74}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"\n{script} failed (exit {result.returncode}). "
                         f"Fix the problem it reported, then re-run "
                         f"`python run_multispecies.py` — finished stages are skipped.")
    print(f"\n[{script} finished in {time.time() - t0:.0f}s]")


def check_raw_data():
    """Fail early and clearly if the genomes are not on disk yet."""
    if not os.path.isdir('data/raw') or not os.listdir('data/raw'):
        raise SystemExit(
            "data/raw/ is empty — the genomes need to be there first, one folder per\n"
            "species, each holding that species' genomic FASTA, protein FASTA and\n"
            "GFF3. See the 'Running the multi-species model' section of README.md.")


def main():
    args = sys.argv[1:]
    force = '--force' in args
    do_scan = '--scan' in args
    domain = '--domain' in args
    start = STAGES.index(args[args.index('--from') + 1]) if '--from' in args else 0

    check_raw_data()

    if do_scan:
        _run('prepare_species.py', ['--scan'])
        print("\nThe manifest is only a draft. Open data/species_manifest.csv and check\n"
              "the annotation_complete column against the data source's own listing of\n"
              "which annotations are finished, then re-run without --scan.")
        return

    if not os.path.exists('data/species_manifest.csv'):
        raise SystemExit(
            "No data/species_manifest.csv. Run `python run_multispecies.py --scan` to\n"
            "draft one from data/raw/, check its annotation_complete column, then\n"
            "re-run this.")

    for stage in STAGES[start:]:
        scripts, marker = STEPS[stage]
        if not force and os.path.exists(marker()):
            print(f"[skip {stage}: {marker()} already exists; --force to redo]")
            continue
        extra = ['--domain'] if (domain and stage == 'train') else []
        for script in scripts:
            _run(script, extra)

    print(f"\n{'=' * 74}")
    print("Done. The result to read is the learned-head margin over group-consensus in")
    print(f"  {P('multispecies_summary.csv')}")
    print("and the label-quality control in")
    print(f"  {P('silver_label_control.csv')}")
    print("A head that does not beat group-consensus has learned only which paralog")
    print("group a protein is in, which the synteny join already told it.")


if __name__ == '__main__':
    main()
