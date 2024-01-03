import json
import time
from threading import Thread

import click

from client import run_client
from datasets.dataset import (
    EdgeFeatureSliceDataset,
    GraphPartitionSliceDataset,
    NodeFeatureSliceDataset,
    PlanetoidDataset,
    PlanetoidDatasetType,
)
from models.graph_attention_network import GAT
from models.graph_convolutional_neural_network import GCN
from server import run_server

# from utils import create_logger

# logger = create_logger(None)


def run_experiment(
    experiement_data: dict[str:any], experiment_name: str = "Bespoke Experiment"
):
    print(f"Running Experiement: {experiment_name}")

    num_clients = experiement_data["num_clients"]
    dataset_name = experiement_data["dataset_name"]
    slice_method = experiement_data["slice_method"]
    percentage_overlap = experiement_data["percentage_overlap"]
    model_type = experiement_data["model_type"]
    num_hidden_params = experiement_data["num_hidden_params"]
    epochs_per_client = experiement_data["epochs_per_client"]
    num_rounds = experiement_data["num_rounds"]

    dataset = PlanetoidDataset(PlanetoidDatasetType(dataset_name))

    if slice_method == "node_feature":
        dataset = NodeFeatureSliceDataset(dataset)
    elif slice_method == "edge_feature":
        dataset = EdgeFeatureSliceDataset(dataset)
    elif slice_method == "graph_partition":
        dataset = GraphPartitionSliceDataset(dataset)

    if model_type == "GAT":
        model = GAT(
            dataset.dataset.num_features,
            num_hidden=num_hidden_params,
            num_classes=dataset.dataset.num_classes,
        )
    elif model_type == "GCN":
        model = GCN(
            dataset.dataset.num_features,
            num_classes=dataset.dataset.num_classes,
        )

    threads = []

    print("Running Server")
    server_thread = Thread(target=run_server, args=(num_rounds,))
    server_thread.start()
    threads.append(server_thread)
    time.sleep(3)

    for client_id in range(0, num_clients):
        print(f"Starting client {client_id} for {experiment_name}")
        client_thread = Thread(
            target=run_client,
            args=(
                model,
                dataset,
                epochs_per_client,
            ),
        )
        client_thread.start()
        threads.append(client_thread)

    for thread in threads:
        thread.join()


@click.command()
@click.option("--num_clients", default=10)
@click.option(
    "--dataset_name",
    default="Cora",
    type=click.Choice(["Cora", "CiteSeer", "PubMed"]),
)
@click.option("--slice_method", default=None)
@click.option("--percentage_overlap", default=0)
@click.option("--model_type", default="GAT", type=click.Choice(["GCN", "GAT"]))
@click.option("--num_hidden_params", default=16)
@click.option("--epochs_per_client", default=10)
@click.option("--num_rounds", default=10)
@click.option("--experiment_config_filename", required=False, default=None)
@click.option("--experiment_name", required=False, default=None)
@click.option("--dry_run", default=True)
def run(
    num_clients: int,
    dataset_name: str,
    slice_method: str,
    percentage_overlap: int,
    model_type: str,
    num_hidden_params: int,
    epochs_per_client: int,
    num_rounds: int,
    experiment_config_filename: str,
    experiment_name: str,
    dry_run: bool,
):
    # open json file
    if experiment_config_filename:
        with open(
            f"experiment_configs/{experiment_config_filename}.json"
        ) as json_file:
            experiments = json.load(json_file)

        if experiment_name:
            experiment_data = experiments[experiment_name]
            print(experiment_data)
            run_experiment(
                experiment_data,
                experiment_name=experiment_name,
            )
        else:
            for experiment_name, experiment_data in experiments.items():
                run_experiment(
                    experiment_data,
                    experiment_name=experiment_name,
                )
    else:
        experiment_data = {
            "num_clients": num_clients,
            "dataset_name": dataset_name,
            "slice_method": slice_method,
            "percentage_overlap": percentage_overlap,
            "model_type": model_type,
            "num_hidden_params": num_hidden_params,
            "epochs_per_client": epochs_per_client,
            "num_rounds": num_rounds,
            "dry_run": dry_run,
        }
        run_experiment(experiement_data=experiment_data)


if __name__ == "__main__":
    run()
