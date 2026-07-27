#!/usr/bin/env python3
"""Submit training and inversions for a completed shared-filesystem dataset."""
from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path

from submit_experiment import (
    AUTOMATION_DIR, PROJECT_DIR, VARIANTS, chunks, frequency_groups,
    markdown_model, number_text, sbatch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="completed data/<experiment-id> directory")
    parser.add_argument("--dry-run", action="store_true", help="print GPU sbatch commands without submitting")
    parser.add_argument("--all-frequency-epochs", type=int, default=140)
    parser.add_argument("--all-frequency-epochs-per-job", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    data_config_path = dataset_dir / "data_config.json"
    train_config_path = dataset_dir / "train_config.json"
    if not data_config_path.is_file() or not train_config_path.is_file():
        raise ValueError("dataset_dir must contain data_config.json and train_config.json")
    data = json.loads(data_config_path.read_text())
    if not (dataset_dir / "valid_data.mat").is_file():
        raise ValueError("valid_data.mat is missing; do not submit GPU work until data generation succeeds")
    expected_files = int(data["ndata"]) // int(data["ndata_per_mat"])
    actual_files = len(list((dataset_dir / "train_data").glob("train_data_*.mat")))
    if actual_files != expected_files:
        raise ValueError(
            "training data is incomplete: expected {} MAT files, found {}".format(expected_files, actual_files)
        )
    start, final, step = float(data["kh_start"]), float(data["kh"]), float(data["dk"])
    count = round((final - start) / step)
    if abs((final - start) / step - count) > 1e-9:
        raise ValueError("dataset frequency range is not divisible by dk")
    frequencies = [start + step * index for index in range(count + 1)]
    if len(frequencies) < 3:
        raise ValueError("at least three frequency steps are required")
    chunk_size = 2
    groups = frequency_groups(frequencies)
    common = {
        "PROJECT_DIR": str(PROJECT_DIR), "AUTOMATION_DIR": str(AUTOMATION_DIR),
        "DATASET_DIR": str(dataset_dir),
    }
    jobs = {}
    # Standard multi-frequency control: all k values are seen from epoch one.
    previous_job = None; all_chunks = []
    for epoch_start in range(0, args.all_frequency_epochs, args.all_frequency_epochs_per_job):
        epoch_end = min(epoch_start + args.all_frequency_epochs_per_job, args.all_frequency_epochs)
        exports = common.copy(); exports.update({"MODEL_NAME":"all-frequencies-mse", "EPOCH_START":str(epoch_start), "EPOCH_END":str(epoch_end)})
        job = sbatch(AUTOMATION_DIR / "training_inversion" / "train_all_frequencies.slurm", exports, previous_job, args.dry_run)
        all_chunks.append({"epoch_start":epoch_start, "epoch_end":epoch_end, "job_id":job}); previous_job = job
    inversion_exports = common.copy(); inversion_exports.update({"MODEL_NAME":"all-frequencies-mse", "INVERSION_OUTPUT_PATH":str(dataset_dir / "all-frequencies-mse" / "inversion_errors.mat")})
    inversion_job = sbatch(AUTOMATION_DIR / "training_inversion" / "run_inversion.slurm", inversion_exports, previous_job, args.dry_run)
    all_dir = dataset_dir / "all-frequencies-mse"; all_dir.mkdir(exist_ok=True)
    (all_dir / "RUN.md").write_text("# Model run: all-frequencies-mse\n\n- Standard multi-frequency baseline: every epoch trains on all k values from the start.\n- All-frequency epochs: {}\n- Epochs per Slurm job: {}\n- Training jobs: {}\n- Inversion job: {}\n".format(args.all_frequency_epochs, args.all_frequency_epochs_per_job, ", ".join(item["job_id"] for item in all_chunks), inversion_job))
    jobs["all-frequencies-mse"] = {"training": all_chunks, "inversion": inversion_job}
    for name, (loss_script, epochs_by_group) in VARIANTS.items():
        previous_job = None
        model_chunks = []
        for group_index, group in enumerate(groups):
            for k_start, k_end in chunks(group, chunk_size):
                exports = common.copy()
                exports.update({
                    "MODEL_NAME": name, "TRAIN_CONFIG_PATH": str(train_config_path),
                    "K_START": str(k_start), "K_END": str(k_end),
                    "EPOCHS": str(epochs_by_group[group_index]),
                    "LOSS_SCRIPT": str(PROJECT_DIR / loss_script),
                    "CHUNK_FILES": "16", "NUM_WORKERS": "8", "SAVE_EVERY_EPOCHS": "25",
                })
                job = sbatch(AUTOMATION_DIR / "training_inversion" / "train_frequency_chunk.slurm", exports, previous_job, args.dry_run)
                model_chunks.append({"k_start": k_start, "k_end": k_end,
                    "kh_start": frequencies[k_start], "kh_end": frequencies[k_end],
                    "epochs": epochs_by_group[group_index], "job_id": job})
                previous_job = job
        inversion_exports = common.copy()
        inversion_exports.update({"MODEL_NAME": name,
            "INVERSION_OUTPUT_PATH": str(dataset_dir / name / "inversion_errors.mat")})
        inversion_job = sbatch(AUTOMATION_DIR / "training_inversion" / "run_inversion.slurm", inversion_exports, previous_job, args.dry_run)
        model = {"loss_script": loss_script, "chunks": model_chunks, "inversion_job_id": inversion_job}
        model_dir = dataset_dir / name
        model_dir.mkdir(exist_ok=True)
        (model_dir / "RUN.md").write_text(markdown_model(name, model, dataset_dir))
        jobs[name] = {"training": [item["job_id"] for item in model_chunks], "inversion": inversion_job}
    (dataset_dir / "gpu_submission.json").write_text(json.dumps({"jobs": jobs, "chunk_size": chunk_size}, indent=2) + "\n")
    print("\nGPU submission record: {}".format(dataset_dir / "gpu_submission.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("error: {}".format(error), file=sys.stderr)
        sys.exit(1)
