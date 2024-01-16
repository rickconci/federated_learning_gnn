import lightning as L
import ray.util.rpdb
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GATConv

from client_utils import get_model_parameters


class GAT(L.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_hidden: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.6,
        num_hidden_layers: int = 0,
        visualise: bool = False,
        learning_rate: float = 0.01,
        global_model_parameters=None,
        proximal_mu: int = 0,
    ):
        super().__init__()
        self.conv1 = GATConv(
            num_features,
            num_hidden,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.learning_rate = learning_rate

        self.hidden_layers = []
        for i in range(num_hidden_layers):
            self.hidden_layers.append(
                GATConv(
                    num_hidden * num_heads,
                    num_hidden,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.conv2 = GATConv(
            num_hidden * num_heads,
            num_classes,
            concat=False,
            heads=1,
            dropout=dropout,
        )

        self.visualise = visualise
        self.criterion = nn.CrossEntropyLoss()
        self.global_model_parameters = global_model_parameters
        self.proximal_mu = proximal_mu

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        for hidden_layer in self.hidden_layers:
            x = F.dropout(x, p=0.6, training=self.training)
            x = hidden_layer(x, edge_index)
            x = F.elu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()

        proximal_regularisation = 0

        if self.proximal_mu != 0:
            proximal_term = 0.0
            for local_weights, global_weights in zip(
                get_model_parameters(self), self.global_model_parameters
            ):
                proximal_term += torch.square(
                    (torch.tensor(local_weights - global_weights)).norm(2)
                )

            proximal_regularisation = (self.proximal_mu / 2) * proximal_term

        loss = (
            self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
            + proximal_regularisation
        )

        self.log(
            "train_loss", loss, prog_bar=False, sync_dist=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage: str):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()

        node_mask = batch.val_mask
        label_mask = batch.val_mask

        if stage == "test":
            node_mask = batch.test_mask
            label_mask = batch.test_mask

        pred = out.argmax(1)

        # loss = self.criterion(out[node_mask], batch.y[label_mask])
        acc = (pred[node_mask] == batch.y[label_mask]).float().mean()

        self.log(
            f"{stage}_accuracy",
            acc,
            prog_bar=False,
            sync_dist=True,
            logger=True,
        )
