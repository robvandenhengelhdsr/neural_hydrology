import os

from pathlib import Path
import shutil
import yaml
import torch
from neural_hydrology.paths import get_env, get_path, load_env
from neural_hydrology.utils.training import (
    get_run_folder_by_name_timestamp,
    load_validated_tensorboard_scalars,
    log_tensorboard_metrics_to_mlflow,
    run_neural_hydrology_model,
)
from neuralhydrology.nh_run import eval_run
import warnings
warnings.filterwarnings("ignore", message="'H' is deprecated and will be removed in a future version")
import mlflow
import numpy as np
import datetime

load_env()
mlflow_uri = get_env("MLFLOW_TRACKING_URI", "databricks")
if mlflow_uri:
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

EXPERIMENT_NAME = "runs" # "LSTM_wonderful_williamson_20260407_124224"
TRIAL_NAME = "trial_28"
PATH_HPO = get_path("HPO_OUTPUT_DIR") / EXPERIMENT_NAME
NUMBER_OF_RETRAININGS = 2

RETRAIN_BASE_DIR = get_path("RETRAIN_BASE_DIR")
DESTINATION_DIR = RETRAIN_BASE_DIR / f"{EXPERIMENT_NAME}_{TRIAL_NAME}"
COPIED_TRIAL_DIR = DESTINATION_DIR / TRIAL_NAME
MLFLOW_EXPERIMENT_NAME = f"/Shared/{EXPERIMENT_NAME}_{TRIAL_NAME}_retrain"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def resolve_source_run_dir(source_trial_dir: Path) -> Path:
    if not source_trial_dir.exists():
        raise RuntimeError(f"Source trial folder does not exist: {source_trial_dir}")
    if not source_trial_dir.is_dir():
        raise RuntimeError(f"Source trial path is not a directory: {source_trial_dir}")

    if (source_trial_dir / "config.yml").exists():
        return source_trial_dir

    subfolders_with_config = [
        f for f in source_trial_dir.iterdir()
        if f.is_dir() and (f / "config.yml").exists()
    ]
    if not subfolders_with_config:
        raise RuntimeError(
            f"No config.yml found in source trial folder or its direct subfolders: {source_trial_dir}"
        )
    if len(subfolders_with_config) > 1:
        raise RuntimeError(
            f"Multiple subfolders with config.yml found in {source_trial_dir}: "
            f"{[f.name for f in subfolders_with_config]}"
        )

    return subfolders_with_config[0]


def copy_trial_folder(source_run_dir: Path, destination_dir: Path) -> Path:
    if destination_dir.exists():
        raise RuntimeError(
            f"Destination folder already exists: {destination_dir}. "
            "Remove it first or change EXPERIMENT_NAME/TRIAL_NAME."
        )

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, destination_dir)
    return destination_dir


def prepare_retrain_config(base_config_path: Path, retrain_dir: Path, i_retrain: int):
    with open(base_config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    seed = i_retrain + 1
    experiment_name = f"{TRIAL_NAME}_retrain_{i_retrain + 1}"
    config["experiment_name"] = experiment_name
    config["run_dir"] = str(retrain_dir)
    config["seed"] = config["seed"] + seed

    for stale_path_key in ["img_log_dir", "train_dir"]:
        config.pop(stale_path_key, None)

    retrain_dir.mkdir(parents=True, exist_ok=True)
    retrain_config_path = retrain_dir / f"config_retrain_{i_retrain + 1}.yml"
    with open(retrain_config_path, "w") as file:
        yaml.dump(config, file)

    return retrain_config_path, experiment_name, seed


def main():
    source_trial_dir = PATH_HPO / TRIAL_NAME
    copied_trial_dir = copy_trial_folder(source_trial_dir, COPIED_TRIAL_DIR)
    copied_run_dir = resolve_source_run_dir(copied_trial_dir)
    copied_config_path = copied_run_dir / "config.yml"

    if not copied_config_path.exists():
        raise RuntimeError(f"Copied config.yml not found: {copied_config_path}")

    with open(config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    print(f"Source trial folder: {source_trial_dir}")
    print(f"Copied trial folder: {copied_trial_dir}")
    print(f"Resolved copied run folder: {copied_run_dir}")
    print(f"Destination folder: {DESTINATION_DIR}")
    print(f"Copied config path: {copied_config_path}")
    print(f"Copied config experiment_name: {config_dict.get('experiment_name')}")
    print(f"Copied config model: {config_dict.get('model')}")

    del config_dict

    with mlflow.start_run(run_name=DESTINATION_DIR.name) as parent_run:
        mlflow.log_params(
            {
                "source_experiment_name": EXPERIMENT_NAME,
                "source_trial_name": TRIAL_NAME,
                "number_of_retrainings": NUMBER_OF_RETRAININGS,
            }
        )

        with mlflow.start_run(run_name=TRIAL_NAME, nested=True):
            mlflow.set_tag("run_type", "copied_original")
            mlflow.log_artifact(str(copied_config_path), artifact_path="config")
            data = load_validated_tensorboard_scalars(copied_run_dir)
            log_tensorboard_metrics_to_mlflow(data, run_folder=copied_run_dir)

        for i_retrain in range(NUMBER_OF_RETRAININGS):
            retrain_dir = DESTINATION_DIR / f"retrain_{i_retrain + 1}"
            config_path, experiment_name, seed = prepare_retrain_config(
                base_config_path=copied_config_path,
                retrain_dir=retrain_dir,
                i_retrain=i_retrain,
            )

            with mlflow.start_run(
                run_name=experiment_name,
                nested=True,
            ):
                mlflow.set_tag("run_type", "retrain")
                mlflow.log_params(
                    {
                        "retrain_index": i_retrain + 1,
                        "experiment_name": experiment_name,
                        "seed": seed,
                    }
                )
                mlflow.log_artifact(str(config_path), artifact_path="config")

                gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
                device_mode = "GPU" if gpu_available else "CPU"
                mlflow.set_tag("device_mode", device_mode)

                run_neural_hydrology_model(config_path)

                run_folder = get_run_folder_by_name_timestamp(
                    trial_dir=retrain_dir,
                    experiment_name=experiment_name,
                )
                print(f"Selected run folder: {Path(run_folder).name}")

                data = load_validated_tensorboard_scalars(run_folder)
                log_tensorboard_metrics_to_mlflow(data, run_folder=run_folder)
                mlflow.log_artifact(str(config_path), artifact_path="config")


if __name__ == "__main__":
    main()
