import argparse
from collections import OrderedDict

import flwr as fl
import lightning as L
import torch

from datasets.dataset import PlanetoidDataset, PlanetoidDatasetType
from models.graph_attention_network import GAT
from models.graph_convolutional_neural_network import GCN

# from datasets.utils.logging import disable_progress_bar


# disable_progress_bar()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: GCN, dataset: PlanetoidDataset):
        self.model = model
        self.dataset = dataset

    def get_parameters(self, config):
        return _get_parameters(self.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = L.Trainer(
            max_epochs=100,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=None,
            accelerator="cpu",
        )
        trainer.fit(
            model=self.model,
            train_dataloaders=self.dataset.train_dataloader(),
            val_dataloaders=self.dataset.val_dataloader(),
        )

        return (
            self.get_parameters(config={}),
            int(self.dataset.dataset[0].train_mask.sum()),
            {},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = L.Trainer(accelerator="cpu")
        results = trainer.test(self.model, self.dataset.test_dataloader())

        loss = results[0]["test_loss"]

        return (
            loss,
            int(self.dataset.dataset[0].test_mask.sum()),
            {"loss": loss},
        )


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition",
    )
    # args = parser.parse_args()
    # node_id = args.node_id

    # Model and data
    dataset = PlanetoidDataset(PlanetoidDatasetType.CORA)
    model = GAT(
        dataset.dataset.num_features,
        num_hidden=16,
        num_classes=dataset.dataset.num_classes,
    )

    # partitioned_data = NodeFeatureSliceDataset(dataset)

    # Flower client]
    client = FlowerClient(model, dataset)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
