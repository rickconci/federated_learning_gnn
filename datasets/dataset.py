from copy import deepcopy
from enum import Enum

import lightning as L
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader.dataloader import DataLoader

from datasets.dataset_info import print_node_feature_slice_dataset_info


class CustomDataset(L.LightningDataModule):
    def __init__(self, dataset: Data = None):
        self.dataset = dataset

    def print_info(self):
        print(f"Number of graphs: {len(self.dataset)}")
        print(f"Number of features: {self.dataset.num_features}")
        print(f"Number of classes: {self.dataset.num_classes}")

        data = self.dataset[0]
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(
            f"Training node "
            f"label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"
        )
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)


class PlanetoidDatasetType(Enum):
    CORA = "Cora"
    CITESEER = "CiteSeer"
    PUBMED = "PubMed"


class PlanetoidDataset:
    def __init__(self, name: PlanetoidDatasetType, num_clients: int):
        self.dataset = Planetoid(root="./datasets", name=name.value)
        self.num_clients = num_clients
        self.num_classes = self.dataset.num_classes
        (
            self.dataset_per_client,
            self.num_features_per_client,
        ) = self._get_datasets()

    def _get_datasets(self) -> None:
        return [
            CustomDataset(dataset=deepcopy(self.dataset))
            for _ in range(self.num_clients)
        ], self.dataset.x.shape[1]


class NodeFeatureSliceDataset2:
    def __init__(
        self,
        name: PlanetoidDatasetType,
        num_clients: int,
        overlap_percent: int = 0,
        verbose: bool = False,
    ) -> None:
        self.dataset = Planetoid(root="./datasets", name=name.value)
        self.num_clients = num_clients
        self.overlap_percent = overlap_percent
        self.verbose = verbose

        self.num_classes = self.dataset.num_classes

        (
            self.dataset_per_client,
            self.num_features_per_client,
        ) = self._get_datasets()

    def _get_datasets(self) -> list[Data]:
        # defining overlap as %  of total dataset size
        data = self.dataset[0]
        features = data.x
        num_features = data.x.shape[1]

        shuffled_indices = torch.randperm(features.size(1))
        shuffled_features = features[:, shuffled_indices]

        num_overlap_features = int(num_features * (self.overlap_percent / 100))
        num_unique_features = num_features - num_overlap_features
        unique_features_per_partition = num_unique_features // self.num_clients
        overlap_features = shuffled_features[:, :num_overlap_features]

        dataset_per_client = []

        for i in range(self.num_clients):
            start_idx = num_overlap_features + i * unique_features_per_partition
            end_idx = start_idx + unique_features_per_partition

            unique_features = shuffled_features[:, start_idx:end_idx]
            partition_features = torch.cat(
                (overlap_features, unique_features), dim=1
            )

            dataset_per_client.append(
                Data(
                    x=partition_features,
                    edge_index=self.dataset.edge_index,
                    y=self.dataset.y,
                    train_mask=self.dataset.train_mask,
                    val_mask=self.dataset.val_mask,
                    test_mask=self.dataset.test_mask,
                    num_classes=self.dataset.num_classes,
                )
            )

        if self.verbose:
            print_node_feature_slice_dataset_info(
                self,
                features,
                partition_features,
                num_features,
                overlap_features,
                num_unique_features,
            )

        return [
            CustomDataset(dataset=[data]) for data in dataset_per_client
        ], partition_features.shape[1]


class NodeFeatureSliceDataset:
    def __init__(
        self,
        name: PlanetoidDatasetType,
        num_clients: int,
        overlap_percent: int = 0,
        verbose: bool = False,
    ) -> None:
        self.dataset = Planetoid(root="./datasets", name=name.value)
        self.num_clients = num_clients
        self.overlap_percent = overlap_percent
        self.verbose = verbose

        self.num_classes = self.dataset.num_classes

        (
            self.dataset_per_client,
            self.num_features_per_client,
        ) = self._get_datasets()

    def _get_datasets(self) -> list[Data]:
        # defining overlap as %  of total dataset size
        data = self.dataset[0]
        node_features = data.x
        num_node_features = data.x.shape[1]

        shuffled_indices = torch.randperm(node_features.size(1))
        shuffled_features = node_features[:, shuffled_indices]
        num_node_features_per_client = num_node_features // self.num_clients
        num_overlap_features = int(
            num_node_features_per_client * (self.overlap_percent / 100)
        )
        num_unique_features = (
            num_node_features_per_client - num_overlap_features
        )
        overlap_features = shuffled_features[:, :num_overlap_features]
        dataset_per_client = []

        for i in range(self.num_clients):
            start_idx = num_overlap_features + i * num_unique_features
            end_idx = start_idx + num_unique_features

            unique_features = shuffled_features[:, start_idx:end_idx]
            partition_features = torch.cat(
                (overlap_features, unique_features), dim=1
            )

            dataset_per_client.append(
                Data(
                    x=partition_features,
                    edge_index=self.dataset.edge_index,
                    y=self.dataset.y,
                    train_mask=self.dataset.train_mask,
                    val_mask=self.dataset.val_mask,
                    test_mask=self.dataset.test_mask,
                    num_classes=self.dataset.num_classes,
                )
            )

        if self.verbose:
            print_node_feature_slice_dataset_info(
                self,
                node_features,
                partition_features,
                num_node_features,
                overlap_features,
                num_unique_features,
            )

        return [
            CustomDataset(dataset=[data]) for data in dataset_per_client
        ], partition_features.shape[1]
