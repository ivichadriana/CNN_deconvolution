import os
import json
import argparse
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import shutil

# Ensure PYTHONPATH includes the parent directory of 'src'
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')
print("Updated PYTHONPATH:", sys.path)  

from src.utils import SimpleMLP, SimpleCNN_3CH, SimpleCNN, PCamDataset, load_training_data, load_testing_data, get_dimensions

script_dir = Path(__file__).resolve().parent

def train_and_evaluate(config, model_type, dataset, image_dim, pcam_data_path=None):

    best_val_loss = float("inf")
    patience = 5  # default value for early stopping

    train_loader, val_loader, shuffle_order_rows1, shuffle_order_columns1 = load_training_data(dataset = dataset,
                                                                                            batch_size=config["batch_size"],
                                                                                            val_split = 0.2,
                                                                                            pcam_data_path=pcam_data_path)

    test_loader, shuffle_order_rows, shuffle_order_columns = load_testing_data(dataset = dataset,
                                                                                            batch_size=config["batch_size"],
                                                                                            pcam_data_path=pcam_data_path,
                                                                                            shuffle_order_rows = shuffle_order_rows1,
                                                                                            shuffle_order_columns = shuffle_order_columns1)
    # Compare shuffle orders for consistency
    if shuffle_order_columns1 is not None and shuffle_order_columns is not None:
        assert np.array_equal(shuffle_order_columns1, shuffle_order_columns), "Shuffle order columns do not match!"
    if shuffle_order_rows1 is not None and shuffle_order_rows is not None:
        assert np.array_equal(shuffle_order_rows1, shuffle_order_rows), "Shuffle order rows do not match!"

    if model_type == "MLP":
        model = SimpleMLP(
            input_dim= image_dim * image_dim,
            fc1_hidden=config["fc1_hidden"],
            fc2_hidden=config["fc2_hidden"],
            fc3_hidden=config["fc3_hidden"]
            )
    elif model_type == 'MLP_3CH':
        model = SimpleMLP(
            input_dim= 3 * image_dim * image_dim,
            fc1_hidden=config["fc1_hidden"],
            fc2_hidden=config["fc2_hidden"],
            fc3_hidden=config["fc3_hidden"]
            )
    elif model_type == 'CNN':
        model = SimpleCNN(
            cha_input=config["cha_input"],
            cha_hidden=config["cha_hidden"],
            fc_hidden=config["fc_hidden"],
            )
    elif model_type == "CNN_3CH":
        model = SimpleCNN_3CH(
            cha_input=config["cha_input"],
            cha_hidden=config["cha_hidden"],
            fc_hidden=config["fc_hidden"]
            )
    else:
        raise ValueError("Invalid model type!")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_losses = []
    val_losses = []
    epochs_without_improvement = 0

    for epoch in range(25):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            if model_type in ["MLP", "MLP_3CH"]:
                inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs only for MLP
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if model_type in ["MLP", "MLP_3CH"]:
                    inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs only for MLP
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))

            # Check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Test phase
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            if model_type in ["MLP", "MLP_3CH"]:
                inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs only for MLP
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall test metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return train_losses, val_losses, accuracy, f1, precision, recall, num_params

def main_train(model_type, dataset, config_path, output_path, pcam_data_path, image_dim):
    # Paths
    config_path = script_dir.parent / f"{config_path}"  # This points to data directory

    out_path = script_dir.parent / f"{output_path}"  # This points to data directory
    os.makedirs(out_path, exist_ok=True)

    config_file_path = os.path.join(config_path, f"best_configs_{model_type}_{dataset}.json")
    results_csv_path = os.path.join(out_path, f"results_{model_type}_{dataset}.csv")
    losses_json_path = os.path.join(out_path, f"losses_{model_type}_{dataset}.json")

    # Load configurations
    with open(config_file_path, "r") as f:
        configurations = json.load(f)

    results = []
    all_losses = {}

    for idx, config in enumerate(configurations):
        
        print(f"Training model {idx + 1}/{len(configurations)}")

        # Train and evaluate
        train_losses, val_losses, accuracy, f1, precision, recall, num_params = train_and_evaluate(config=config, 
                                                                                                    model_type=model_type, 
                                                                                                    dataset=dataset, 
                                                                                                    image_dim=image_dim, 
                                                                                                    pcam_data_path=pcam_data_path)

        # Store results
        results.append({
            "model_index": idx,
            **config,
            "num_params": num_params,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        })

        # Store losses
        all_losses[f"model_{idx}"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        # Save intermediate results
        with open(results_csv_path, "a") as f:
            pd.DataFrame([results[-1]]).to_csv(f, header=f.tell() == 0, index=False)

        with open(losses_json_path, "w") as f:
            json.dump(all_losses, f, indent=4)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # Save losses to JSON
    with open(losses_json_path, "w") as f:
        json.dump(all_losses, f, indent=4)
    print(f"Losses saved to {losses_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training for models according to configurations.")
    parser.add_argument("--model_type", type=str, choices=["MLP", "MLP_3CH", "CNN", "CNN_3CH"], required=True, help="Type of model (MLP, MLP_3CH, CNN).")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "MNISTshuffled", "FashMNIST", "FashMNISTshuffled", "CIFAR10shuffled", "CIFAR10", "PCam", "PCamshuffled"], required=True, help="Dataset to use = NAME + shuffled.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the best configurations.")
    parser.add_argument("--pcam_data_path", type=str, required=True, help="Path to the PCam data.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the tuned configurations.")

    args = parser.parse_args()

    image_dim = get_dimensions(args.dataset)

    main_train(model_type=args.model_type, 
                dataset=args.dataset,
                output_path=args.output_path, 
                config_path=args.config_path, 
                pcam_data_path=args.pcam_data_path, 
                image_dim=image_dim)