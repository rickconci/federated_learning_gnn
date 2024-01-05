import lightning as L
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GATConv


class GAT(L.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_hidden: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.6,
        visualise: bool = False,
    ):
        super().__init__()

        self.conv1 = GATConv(
            num_features, num_hidden, heads=num_heads, dropout=dropout
        )
        self.conv2 = GATConv(
            num_hidden * num_heads,
            num_classes,
            concat=False,
            heads=1,
            dropout=dropout,
        )

        self.visualise = visualise
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])

        self.log(
            f"train_loss",
            loss,
            prog_bar=False,
            sync_dist=True,
            logger=True,
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
            f"{stage}_loss", acc, prog_bar=False, sync_dist=True, logger=True
        )
