import os
import json
import ray
import torch
import sys
import argparse
import shutil
import random
from pathlib import Path
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Ensure PYTHONPATH includes the parent directory of 'src'
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')
print("Updated PYTHONPATH:", sys.path)

from src.utils import (
    SimpleMLP,
    SimpleCNN_3CH,
    SimpleTransformer,
    SimpleCNN,
    load_training_data_fullshuffle,
    get_dimensions,
)

script_dir = Path(__file__).resolve().parent


def train_model(config, model_type, dataset, image_dim, pcam_data_path=None):
    train_loader, val_loader, _ = load_training_data_fullshuffle(
        dataset=dataset,
        batch_size=config["batch_size"],
        val_split=0.2,
        pcam_data_path=pcam_data_path
    )

    if model_type in ["MLP", "MLP_3CH"]:
        if "CIFAR10" in dataset or "PCam" in dataset:
            in_channels = 3
            model = SimpleMLP(
                input_dim=in_channels * image_dim * image_dim,
                fc1_hidden=config["fc1_hidden"],
                fc2_hidden=config["fc2_hidden"],
                fc3_hidden=config["fc3_hidden"],
                dropout=config["dropout"]
            )
        elif "MNIST" in dataset:
            model = SimpleMLP(
                input_dim=image_dim * image_dim,
                fc1_hidden=config["fc1_hidden"],
                fc2_hidden=config["fc2_hidden"],
                fc3_hidden=config["fc3_hidden"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

    elif model_type in ["CNN", "CNN_3CH"]:
        if "CIFAR10" in dataset or "PCam" in dataset:
            model = SimpleCNN_3CH(
                cha_input=config["cha_input"],
                cha_hidden=config["cha_hidden"],
                fc_hidden=config["fc_hidden"],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
                dropout=config["dropout"]
            )
        elif "MNIST" in dataset:
            model = SimpleCNN(
                cha_input=config["cha_input"],
                cha_hidden=config["cha_hidden"],
                fc_hidden=config["fc_hidden"],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

    elif model_type in ["TRANS", "TRANS_3CH"]:
        if "CIFAR10" in dataset or "PCam" in dataset:
            model = SimpleTransformer(
                image_size=image_dim,
                patch_size=config["patch_size"],
                in_channels=3,
                emb_dim=config["emb_dim"],
                num_heads=config["num_heads"],
                mlp_dim=config["mlp_dim"],
                dropout=config["dropout"]
            )
        elif "MNIST" in dataset:
            model = SimpleTransformer(
                image_size=image_dim,
                patch_size=config["patch_size"],
                in_channels=1,
                emb_dim=config["emb_dim"],
                num_heads=config["num_heads"],
                mlp_dim=config["mlp_dim"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(25):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        accuracy = correct / total
        tune.report(val_loss=val_loss, accuracy=accuracy)


def run_tuning(
    model_type,
    dataset,
    config,
    output_path,
    tmp_dir,
    working_dir,
    num_iterations,
    pcam_data_path=None,
    image_dim=28
):
    ray.shutdown()

    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "py_modules": [str(script_dir.parent / "src")],
        },
        _temp_dir=tmp_dir,
        log_to_driver=False,
        include_dashboard=False
    )

    scheduler = ASHAScheduler(
        max_t=25,
        grace_period=5,
        metric="val_loss",
        mode="min"
    )

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_iterations):
        experiment_name = f"tuning_iteration_{i+1}"
        print(f"Running iteration number {i+1}...")

        analysis = tune.run(
            tune.with_parameters(
                train_model,
                model_type=model_type,
                dataset=dataset,
                image_dim=image_dim,
                pcam_data_path=pcam_data_path
            ),
            config=config,
            num_samples=100,
            scheduler=scheduler,
            resources_per_trial={"cpu": 8},
            fail_fast=False,
            storage_path=tmp_dir,
            name=experiment_name
        )

        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        print(f"Best hyperparameters for iteration {i+1}: {best_config}")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                with open(output_path, "r") as f:
                    existing_configs = json.load(f)
            except (json.JSONDecodeError, IOError):
                print("Warning: Unable to read existing configurations. Starting fresh.")
                existing_configs = []
        else:
            existing_configs = []

        existing_configs.append(best_config)

        try:
            with open(output_path, "w") as f:
                json.dump(existing_configs, f, indent=4)
            print(f"Configuration for iteration {i+1} saved successfully.")
        except IOError as e:
            print(f"Error saving configuration for iteration {i+1}: {e}")

        try:
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to clean up tmp_dir: {e}")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for models.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["MLP", "MLP_3CH", "CNN", "CNN_3CH", "TRANS", "TRANS_3CH"],
        required=True,
        help="Type of model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "MNIST",
            "MNISTshuffled",
            "FashMNIST",
            "FashMNISTshuffled",
            "CIFAR10shuffled",
            "CIFAR10",
            "PCam",
            "PCamshuffled"
        ],
        required=True,
        help="Dataset to use."
    )
    parser.add_argument("--tmp_dir", type=str, required=True, help="Temporary directory for Ray Tune.")
    parser.add_argument("--working_dir", type=str, required=True, help="Working directory for Ray Tune.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the best configurations.")
    parser.add_argument("--pcam_data_path", type=str, required=True, help="Path to the PCam data.")
    parser.add_argument("--num_iterations", type=str, required=True, help="How many times to tune 100 samples.")

    args = parser.parse_args()

    os.environ["RAY_TMPDIR"] = args.tmp_dir

    config = {
        "MLP": {
            "fc1_hidden": tune.randint(196, 693),
            "fc2_hidden": tune.randint(130, 686),
            "fc3_hidden": tune.randint(98, 272),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120]),
            "dropout": tune.quniform(0.1, 0.6, 0.1),
        },

        "MLP_3CH": {
            "fc1_hidden": tune.randint(63, 416),
            "fc2_hidden": tune.randint(42, 316),
            "fc3_hidden": tune.randint(98, 316),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120]),
            "dropout": tune.quniform(0.1, 0.6, 0.1),
        },

        "CNN": {
            "cha_input": tune.randint(56, 87),
            "cha_hidden": tune.randint(88, 148),
            "fc_hidden": tune.randint(98, 272),
            "kernel_size": tune.choice([3, 5]),
            "stride": tune.choice([1, 2]),
            "padding": tune.choice([0, 1]),
            "dropout": tune.quniform(0.1, 0.6, 0.1),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120]),
        },

        "CNN_3CH": {
            "cha_input": tune.randint(78, 212),
            "cha_hidden": tune.randint(68, 120),
            "fc_hidden": tune.randint(98, 273),
            "kernel_size": tune.choice([3, 5]),
            "stride": tune.choice([1, 2]),
            "padding": tune.choice([0, 1]),
            "dropout": tune.quniform(0.1, 0.6, 0.1),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120]),
        },

        "TRANS": {
            "patch_size": tune.choice([2, 4]),
            "num_heads": tune.choice([1, 2]),
            "emb_dim": tune.sample_from(
                lambda spec: random.choice([
                    e for e in range(102, 249)
                    if e % spec.config.num_heads == 0
                ])
            ),
            "mlp_dim": tune.randint(98, 271),
            "dropout": tune.quniform(0.1, 0.6, 0.1),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120]),
        },

        "TRANS_3CH": {
            "patch_size": tune.choice([2, 4]),
            "num_heads": tune.choice([1, 2]),
            "emb_dim": tune.sample_from(
                lambda spec: random.choice([
                    e for e in range(102, 284)
                    if e % spec.config.num_heads == 0
                ])
            ),
            "mlp_dim": tune.randint(98, 271),
            "dropout": tune.quniform(0.1, 0.6, 0.1),
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 120]),
        },
    }[args.model_type]

    image_dim = get_dimensions(args.dataset)
    num_iterations = int(args.num_iterations)

    run_tuning(
        model_type=args.model_type,
        dataset=args.dataset,
        config=config,
        output_path=args.output_path,
        tmp_dir=args.tmp_dir,
        working_dir=args.working_dir,
        pcam_data_path=args.pcam_data_path,
        image_dim=image_dim,
        num_iterations=num_iterations
    )