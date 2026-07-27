#!/usr/bin/env python3
"""Create and submit one complete nc=20 curriculum-learning experiment.

This program runs briefly on the Slurm login node.  It never waits for work:
Slurm dependencies sequence the data array, model-specific training chunks,
and inversions after this program exits.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


AUTOMATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AUTOMATION_DIR.parent
VARIANTS = {
    "equal-mse": ("chunk_train2.py", [140, 140, 140]),
    "equal-new-loss": ("chunk_train_new_loss.py", [140, 140, 140]),
    "up-mse": ("chunk_train2.py", [60, 120, 240]),
    "down-mse": ("chunk_train2.py", [240, 120, 60]),
    "up-new-loss": ("chunk_train_new_loss.py", [60, 120, 240]),
    "down-new-loss": ("chunk_train_new_loss.py", [240, 120, 60]),
}
INVERSION_DEFAULTS = {
    "noise_levels": [0, 0.25, 0.5, 0.75, 1],
    "num_tweaks": 100,
    "tweak_factor": 0.025,
}
DRY_JOB_COUNTER = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="experiment JSON file")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_DIR / "data",
        help="parent directory for experiment datasets (default: ./data)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write the experiment record and print sbatch commands without submitting",
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="submit only the CPU data-generation array; use submit_gpu_pipeline.py after it finishes",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_config_path(value: str | None, standard_name: str) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value))
    candidates.extend((PROJECT_DIR / "configs" / standard_name, PROJECT_DIR / standard_name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Could not find {standard_name}; searched: {searched}")


def number_text(value: float | int) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else format(value, "g").replace(".", "p")


def validate_and_frequencies(config: dict[str, Any]) -> list[float]:
    required = ("ndata", "generation_noise", "kh_final", "dk")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required experiment settings: {', '.join(missing)}")

    start = float(config.get("kh_start", 1))
    final = float(config["kh_final"])
    step = float(config["dk"])
    if start <= 0 or final < start or step <= 0:
        raise ValueError("kh_start and dk must be positive, and kh_final must be >= kh_start")
    count_float = (final - start) / step
    count = round(count_float)
    if abs(count_float - count) > 1e-9:
        raise ValueError("(kh_final - kh_start) must be exactly divisible by dk")
    if int(config["ndata"]) <= 0:
        raise ValueError("ndata must be positive")
    if int(config.get("training_chunk_size", 2)) <= 0:
        raise ValueError("training_chunk_size must be positive")
    frequencies = [start + step * index for index in range(count + 1)]
    if len(frequencies) < 3:
        raise ValueError("At least three frequency steps are required for the three-group schedule")
    return frequencies


def frequency_groups(frequencies: list[float]) -> list[list[int]]:
    """Return three balanced groups as zero-based indices, excess at high k."""
    n = len(frequencies)
    base = n // 3
    sizes = [base, base, n - 2 * base]
    groups: list[list[int]] = []
    position = 0
    for size in sizes:
        groups.append(list(range(position, position + size)))
        position += size
    return groups


def chunks(indices: list[int], size: int) -> list[tuple[int, int]]:
    return [(part[0], part[-1]) for part in (indices[i:i + size] for i in range(0, len(indices), size)) if part]


def experiment_id(data_config: dict[str, Any], timestamp: str, suffix: str | None) -> str:
    parts = [
        f"star{data_config['nc']}",
        f"kh{number_text(data_config['kh_start'])}",
        number_text(float(data_config['dk']) * 100),
        number_text(data_config['kh']),
        f"n{data_config['n_tgt']}",
        number_text(data_config['ndata']),
        f"noise{number_text(float(data_config['noise_lvl']) * 100)}",
    ]
    safe_suffix = re.sub(r"[^A-Za-z0-9_-]+", "-", suffix or "").strip("-")
    return "_".join(parts) + f"__{safe_suffix + '_' if safe_suffix else ''}{timestamp}"


def sbatch(script: Path, exports: dict[str, str], dependency: str | None, dry_run: bool) -> str | None:
    global DRY_JOB_COUNTER
    export_text = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    command = ["sbatch", "--parsable", "--chdir", str(PROJECT_DIR), "--export", export_text]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script))
    print(shlex.join(command))
    if dry_run:
        DRY_JOB_COUNTER += 1
        return f"DRY{DRY_JOB_COUNTER:03d}"
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";", 1)[0]


def markdown_experiment(manifest: dict[str, Any]) -> str:
    data = manifest["data"]
    lines = [
        f"# Experiment {manifest['experiment_id']}", "",
        f"Submitted: {manifest['submitted_at']}", "",
        "## Data generation", "",
        f"- Dataset directory: `{manifest['dataset_dir']}`",
        f"- `nc`: {data['nc']}",
        f"- `ndata`: {data['ndata']}",
        f"- Generation noise: {data['noise_lvl']}",
        f"- Frequencies: {', '.join(number_text(value) for value in manifest['frequencies'])}",
        f"- Training chunk size: {manifest['training_chunk_size']} frequency steps",
        "", "## Frequency groups", "",
    ]
    for index, group in enumerate(manifest["groups"], start=1):
        display = group["frequencies"]
        lines.append(f"- Group {index}: {number_text(display[0])}–{number_text(display[-1])}")
    lines += ["", "## Fixed inversion benchmark", ""]
    lines += [f"- Noise levels: {', '.join(map(str, INVERSION_DEFAULTS['noise_levels']))}"]
    lines += [f"- Tweaks per noise level: {INVERSION_DEFAULTS['num_tweaks']}"]
    lines += [f"- Tweak factor: {INVERSION_DEFAULTS['tweak_factor']}"]
    lines += ["", "## Slurm jobs", "", f"- Data generation array: `{manifest['jobs']['data_generation']}`"]
    for name, job in manifest["jobs"]["models"].items():
        lines.append(f"- `{name}`: training `{', '.join(job['training'])}`; inversion `{job['inversion']}`")
    return "\n".join(lines) + "\n"


def markdown_model(name: str, model: dict[str, Any], dataset_dir: Path) -> str:
    lines = [f"# Model run: {name}", "", f"- Dataset: `{dataset_dir}`", f"- Loss script: `{model['loss_script']}`", "", "## Training chunks", ""]
    for chunk in model["chunks"]:
        lines.append(
            f"- frequency indices {chunk['k_start']}–{chunk['k_end']} "
            f"(k={number_text(chunk['kh_start'])}–{number_text(chunk['kh_end'])}), "
            f"{chunk['epochs']} epochs; Slurm job `{chunk['job_id']}`"
        )
    lines += ["", f"- Inversion job: `{model['inversion_job_id']}`", "- Aggregate result: `inversion_errors.mat`", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    user_config = load_json(args.config.resolve())
    frequencies = validate_and_frequencies(user_config)
    data_source = resolve_config_path(user_config.get("data_config_path"), "nc20.json")
    train_source = resolve_config_path(user_config.get("train_config_path"), "train_nc20.json")
    data_config = load_json(data_source)
    data_config.update({
        "nc": 20,
        "ndata": int(user_config["ndata"]),
        "noise_lvl": float(user_config["generation_noise"]),
        "kh_start": float(user_config.get("kh_start", 1)),
        "kh": float(user_config["kh_final"]),
        "dk": float(user_config["dk"]),
    })
    data_config["nk"] = len(frequencies)
    chunk_size = int(user_config.get("training_chunk_size", 2))
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = experiment_id(data_config, timestamp, user_config.get("run_label"))
    dataset_dir = (args.output_root.resolve() / run_id)
    if dataset_dir.exists():
        raise FileExistsError(f"Refusing to reuse an existing experiment directory: {dataset_dir}")

    dataset_dir.mkdir(parents=True)
    (dataset_dir / "status").mkdir()
    (PROJECT_DIR / "logs" / "pipeline").mkdir(parents=True, exist_ok=True)
    generated_data_config = dataset_dir / "data_config.json"
    copied_train_config = dataset_dir / "train_config.json"
    generated_data_config.write_text(json.dumps(data_config, indent=2) + "\n")
    shutil.copy2(train_source, copied_train_config)

    groups = frequency_groups(frequencies)
    manifest: dict[str, Any] = {
        "experiment_id": run_id,
        "submitted_at": datetime.now().astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "data": data_config,
        "data_config_source": str(data_source),
        "train_config_source": str(train_source),
        "frequencies": frequencies,
        "training_chunk_size": chunk_size,
        "inversion": INVERSION_DEFAULTS,
        "groups": [
            {"indices": group, "frequencies": [frequencies[index] for index in group]}
            for group in groups
        ],
        "jobs": {"data_generation": None, "models": {}},
    }
    common_exports = {
        "PROJECT_DIR": str(PROJECT_DIR),
        "AUTOMATION_DIR": str(AUTOMATION_DIR),
        "DATASET_DIR": str(dataset_dir),
    }
    data_job = sbatch(
        AUTOMATION_DIR / "data_generation" / "generate_data_array.slurm",
        common_exports | {"DATA_CONFIG_PATH": str(generated_data_config)},
        dependency=None,
        dry_run=args.dry_run,
    )
    manifest["jobs"]["data_generation"] = data_job

    if args.data_only:
        (dataset_dir / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (dataset_dir / "EXPERIMENT.md").write_text(markdown_experiment(manifest))
        print(f"\nData-only experiment record: {dataset_dir / 'EXPERIMENT.md'}")
        return 0

    for name, (loss_script, epochs_by_group) in VARIANTS.items():
        previous_job = data_job
        model_chunks: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups):
            for k_start, k_end in chunks(group, chunk_size):
                exports = common_exports | {
                    "MODEL_NAME": name,
                    "TRAIN_CONFIG_PATH": str(copied_train_config),
                    "K_START": str(k_start),
                    "K_END": str(k_end),
                    "EPOCHS": str(epochs_by_group[group_index]),
                    "LOSS_SCRIPT": str(PROJECT_DIR / loss_script),
                    "CHUNK_FILES": str(user_config.get("chunk_files", 16)),
                    "NUM_WORKERS": str(user_config.get("num_workers", 8)),
                    "SAVE_EVERY_EPOCHS": str(user_config.get("save_every_epochs", 25)),
                }
                job = sbatch(AUTOMATION_DIR / "training_inversion" / "train_frequency_chunk.slurm", exports, previous_job, args.dry_run)
                model_chunks.append({
                    "k_start": k_start, "k_end": k_end,
                    "kh_start": frequencies[k_start], "kh_end": frequencies[k_end],
                    "epochs": epochs_by_group[group_index], "job_id": job,
                })
                previous_job = job
        inversion_job = sbatch(
            AUTOMATION_DIR / "training_inversion" / "run_inversion.slurm",
            common_exports | {
                "MODEL_NAME": name,
                "INVERSION_OUTPUT_PATH": str(dataset_dir / name / "inversion_errors.mat"),
            },
            previous_job,
            args.dry_run,
        )
        model = {"loss_script": loss_script, "chunks": model_chunks, "inversion_job_id": inversion_job}
        manifest["jobs"]["models"][name] = {
            "training": [chunk["job_id"] for chunk in model_chunks],
            "inversion": model["inversion_job_id"],
        }
        model_dir = dataset_dir / name
        model_dir.mkdir()
        (model_dir / "RUN.md").write_text(markdown_model(name, model, dataset_dir))

    (dataset_dir / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dataset_dir / "EXPERIMENT.md").write_text(markdown_experiment(manifest))
    print(f"\nExperiment record: {dataset_dir / 'EXPERIMENT.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
