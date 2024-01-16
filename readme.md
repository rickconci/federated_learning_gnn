# Federated Learning GNN

## How to run experiment

1. Setup any virtual (i.e. with `conda` or `virtualenv`) with a `python3.11` interpreter
2. Install dependencies under `requirements.txt`
3. To perform a single federated learning experiment, execute the `run.py` file with the following arguments:

```python
@click.command()
@click.option("--num_clients", default=10)
@click.option(
    "--dataset_name",
    default="Cora",
    type=click.Choice(["Cora", "CiteSeer", "PubMed"]),
)
@click.option(
    "--slice_method",
    default=None,
    type=click.Choice([None, "node_feature", "node_feature2"]),
)
@click.option("--percentage_overlap", default=0)
@click.option("--model_type", default="GAT", type=click.Choice(["GCN", "GAT"]))
@click.option("--num_hidden_params", default=16)
@click.option("--num_hidden_layers", default=1)
@click.option("--learning_rate", default=0.01)
@click.option("--epochs_per_client", default=10)
@click.option("--num_rounds", default=10)
@click.option("--aggregation_strategy", default="FedAvg")
@click.option("--experiment_config_filename", required=False, default=None)
@click.option("--experiment_name", required=False, default=None)
@click.option("--dry_run", default=True)
```

Example usage:

```shell
python3.11 run.py --num_clients 10 --dataset_name Cora --slice_method node_feature --percentage_overlap 30 --model_type GAT
```

Please note to perform data poisoning experiments, please checkout to the `data_poisoning` branch.