import flwr as fl
import argparse


def main() -> None:

    parser = argparse.ArgumentParser(description="Flower Server configuration")
    parser.add_argument(
        '--Fed_Rounds', 
        type=int, 
        required=True, 
        help='Number of Federated Rounds')
    args = parser.parse_args()



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
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.Fed_Rounds),
        strategy=strategy,
    )



if __name__ == "__main__":
    main()
