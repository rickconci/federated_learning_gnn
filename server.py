import flwr as fl


def run_server(num_federated_rounds: int) -> None:
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=num_federated_rounds),
        strategy=strategy,
    )
