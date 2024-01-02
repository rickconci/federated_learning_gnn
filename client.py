import argparse
from collections import OrderedDict

import flwr as fl
import lightning as L
import torch

from datasets.dataset import PlanetoidDataset, PlanetoidDatasetType
from datasets.dataset import NodeFeatureSliceDataset, EdgeFeatureSliceDataset, GraphPartitionSliceDataset

from models.graph_attention_network import GAT
from models.graph_convolutional_neural_network import GCN

# from datasets.utils.logging import disable_progress_bar


# disable_progress_bar()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: GCN, dataset: PlanetoidDataset, epochs: int):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs

    def get_parameters(self, config):
        return _get_parameters(self.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = L.Trainer(
            max_epochs=self.epochs,
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
    parser = argparse.ArgumentParser(description="Flower Client configuration")

    parser.add_argument(
        "--num_clients",
        type=int,
        required=True,
        help="Total number of clients being created",
    )

    ## FEDERATION ARGUMENTS
    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
        help="Identifies specific client being created",
    )

    ## DATASET ARGUMENTS
    parser.add_argument(
        "--dataset_choice",
        type=str,
        required=True,
        choices=['Cora', 'Citeseer', 'Pubmed'],
        help="Choice of dataset"
    )
    
    parser.add_argument(
        "--slice_method",
        type=str,
        required=True,
        choices=['None', 'node_feature', 'edge_feature', 'graph_partition'],
        help="Method used for slicing the data"
    )

    parser.add_argument(
        "--num_overlap", 
        type=float, 
        required= False,  #CHANGE TO TRUE
        choices = range(0,100),
        default = 0,
        help="Percentage data overlap across clients"
    )

    ## MODEL ARGUMENTS
    parser.add_argument(
         "--GNN_model", 
        type=str, 
        required=True, 
        default = 'GAT',
        choices = ['GAT', 'GCN'],
        help="type of model"
    )

    parser.add_argument(
         "--GNN_hidden", 
        type=int, 
        required=False,  #Change to true?
        default = 16, 
        help="number of hidden layers in model"
    )

    ## FEDERATED TRAINING ARGUMENTS
    parser.add_argument(
         "--Epochs_per_client", 
        type=int, 
        required=True,  #Change to true?
        default = 100, 
        help="Number of training epochs per client"
    )

    args = parser.parse_args()

    # DATASET CHOICE
    if args.dataset_choice=="Cora":
        dataset = PlanetoidDataset(PlanetoidDatasetType.CORA)
    elif args.dataset_choice=="Pubmed":
        dataset = PlanetoidDataset(PlanetoidDatasetType.PUBMED)
    elif args.dataset_choice=="Citeseer":
        dataset = PlanetoidDataset(PlanetoidDatasetType.CITESEER)

    # SLICING METHOD + OVERLAP
    if args.slice_method =="node_feature":
        node_slicer = NodeFeatureSliceDataset(dataset) #input dataset to node feat slicer
        sliced_data = node_slicer.slice_dataset() #input overlap choice in node feat slicer + total num_nodes + node_ID
    elif args.slice_method =="edge_feature":
        edge_slicer = EdgeFeatureSliceDataset(dataset)
        sliced_data = edge_slicer.slice_dataset()
    elif args.slice_method=="graph_partition":
        graph_slicer = GraphPartitionSliceDataset(dataset)
        sliced_data = graph_slicer.slice_dataset()  #LOTS more inputs here
    else:
        sliced_data = dataset
    ## Make sure sliced_data is in same format as dataset.dataset to be inputted to model in same format?
        
    if args.GNN_model =="GAT":
        model = GAT(
            sliced_data.dataset.num_features,
            num_hidden=args.GNN_hidden,
            num_classes=dataset.dataset.num_classes,
        )
    elif args.GNN_model =="GCN":
        model = GCN(
            sliced_data.dataset.num_features,
            #num_hidden=args.GNN_hidden,
            num_classes=dataset.dataset.num_classes,
        )

    # Flower client
    client = FlowerClient(model, sliced_data, args.Epochs_per_client)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
