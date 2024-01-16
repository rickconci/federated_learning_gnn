import itertools
import json

from constants import (
    gat_citeseer_constants,
    gat_cora_constants,
    gat_pubmed_constants,
)

fixed_params = {
    "model_type": "GAT",
    "epochs_per_client": 10,
    "num_rounds": 50,
    "dry_run": False,
}

datasets = ["Cora", "CiteSeer", "PubMed"]

aggregation_strategies = [
    "FedAvg",
    "FedProx",
    "FedYogi",
    "FedAdam",
    "FedOpt",
    "FedAdagrad",
]

num_clients = [2, 4, 8, 10]
percentage_overlap = [0, 25, 50, 100]
grid_search = list(
    itertools.product(num_clients, percentage_overlap, aggregation_strategies)
)

for slice_method in ["node_feature", "node_feature2"]:
    for dataset in datasets:
        new_params = fixed_params.copy()
        new_params["slice_method"] = slice_method
        all_experiments = {}
        new_params["dataset_name"] = dataset
        if dataset == "Cora":
            new_params.update(gat_cora_constants)

        elif dataset == "CiteSeer":
            new_params.update(gat_citeseer_constants)

        elif dataset == "PubMed":
            new_params.update(gat_pubmed_constants)
        for num_client, percentage_overlap, aggregation_strategy in grid_search:
            new_params["num_clients"] = num_client
            new_params["percentage_overlap"] = percentage_overlap
            new_params["aggregation_strategy"] = aggregation_strategy
            all_experiments[
                f"GAT-num_clients_{num_client}-percentage_overlap_{percentage_overlap}-aggregation_strategy_{aggregation_strategy}"
            ] = new_params.copy()

        with open(
            f"experiment_configs/GAT_{dataset}_overlap_experiments_slice_method_{slice_method}_rev2.json",
            "w",
        ) as outfile:
            json.dump(all_experiments, outfile, indent=4)
