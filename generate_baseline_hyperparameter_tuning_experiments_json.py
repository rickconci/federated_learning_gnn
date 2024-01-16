import itertools
import json

num_features = [4, 8, 16]

num_hidden_layers = [0, 1, 2]

learning_rates = [0.01, 0.001, 0.0001]

grid_search = list(
    itertools.product(num_features, num_hidden_layers, learning_rates)
)


fixed_params = {
    "num_clients": 1,
    # "dataset_name": "Cora",
    "slice_method": "node_feature",
    "percentage_overlap": 0,
    "model_type": "GAT",
    "epochs_per_client": 1,
    "num_rounds": 50,
    "aggregation_strategy": "FedAvg",
    "dry_run": False,
}
for dataset_n in ["Cora", "PubMed", "CiteSeer"]:
    for model_type in ["GAT", "GCN"]:
        all_experiments = {}
        new_params = fixed_params.copy()
        new_params["model_type"] = model_type
        new_params["dataset_name"] = dataset_n

        for features, hidden_layers, learning_rate in grid_search:
            new_params_w_l_f = new_params.copy()
            new_params_w_l_f["num_hidden_params"] = features
            new_params_w_l_f["num_hidden_layers"] = hidden_layers
            new_params_w_l_f["learning_rate"] = learning_rate

            all_experiments[
                f"{model_type}_features-{features}_layers-{hidden_layers}_lr-{learning_rate}"
            ] = new_params_w_l_f

        with open(
            f"experiment_configs/{model_type}_{dataset_n}_hyperparameter_tuning_experiments.json",
            "w",
        ) as outfile:
            json.dump(all_experiments, outfile, indent=4)
