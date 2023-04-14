import pytorch_lightning as pl
import torch

from src.const import ZINC_TRAIN_LINKER_ID2SIZE, ZINC_TRAIN_LINKER_SIZE2ID
from src.linker_size import SizeGNN
from src.egnn import coord2diff
from src.datasets import ZincDataset, get_dataloader, collate_with_fragment_edges
from typing import Dict, List, Optional
from torch.nn.functional import cross_entropy, mse_loss, sigmoid

from pdb import set_trace


class SizeClassifier(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    metrics: Dict[str, List[float]] = {}

    def __init__(
            self, data_path, train_data_prefix, val_data_prefix,
            in_node_nf, hidden_nf, out_node_nf, n_layers, batch_size, lr, torch_device,
            normalization=None,
            loss_weights=None,
            min_linker_size=None,
            linker_size2id=ZINC_TRAIN_LINKER_SIZE2ID,
            linker_id2size=ZINC_TRAIN_LINKER_ID2SIZE,
            task='classification',
    ):
        super(SizeClassifier, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.min_linker_size = min_linker_size
        self.linker_size2id = linker_size2id
        self.linker_id2size = linker_id2size
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_weights = None if loss_weights is None else torch.tensor(loss_weights, device=self.torch_device)
        self.gnn = SizeGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            n_layers=n_layers,
            device=self.torch_device,
            normalization=normalization,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device
            )
            self.val_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        elif stage == 'val':
            self.val_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_with_fragment_edges, shuffle=True)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_with_fragment_edges)

    def test_dataloader(self):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_with_fragment_edges)

    def forward(self, data, return_loss=True, with_pocket=False):
        h = data['one_hot']
        x = data['positions']
        fragment_mask = data['fragment_only_mask'] if with_pocket else data['fragment_mask']
        linker_mask = data['linker_mask']
        edge_mask = data['edge_mask']
        edges = data['edges']

        # Considering only fragments
        x = x * fragment_mask
        h = h * fragment_mask

        # Reshaping
        bs, n_nodes = x.shape[0], x.shape[1]
        fragment_mask = fragment_mask.view(bs * n_nodes, 1)
        x = x.view(bs * n_nodes, -1)
        h = h.view(bs * n_nodes, -1)

        # Prediction
        distances, _ = coord2diff(x, edges)
        distance_edge_mask = (edge_mask.bool() & (distances < 6)).long()
        output = self.gnn.forward(h, edges, distances, fragment_mask, distance_edge_mask)
        output = output.view(bs, n_nodes, -1).mean(1)

        if return_loss:
            true = self.get_true_labels(linker_mask)
            loss = cross_entropy(output, true, weight=self.loss_weights)
        else:
            loss = None

        return output, loss

    def get_true_labels(self, linker_mask):
        labels = []
        sizes = linker_mask.squeeze().sum(-1).long().detach().cpu().numpy()
        for size in sizes:
            label = self.linker_size2id.get(size)
            if label is None:
                label = self.linker_size2id[max(self.linker_id2size)]
            labels.append(label)
        labels = torch.tensor(labels, device=linker_mask.device, dtype=torch.long)
        return labels

    def training_step(self, data, *args):
        _, loss = self.forward(data)
        return {'loss': loss}

    def validation_step(self, data, *args):
        _, loss = self.forward(data)
        return {'loss': loss}

    def test_step(self, data, *args):
        loss = self.forward(data)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        correct = 0
        total = 0
        for data in self.val_dataloader():
            output, _ = self.forward(data)
            pred = output.argmax(dim=-1)
            true = self.get_true_labels(data['linker_mask'])
            correct += (pred == true).sum()
            total += len(pred)

        accuracy = correct / total
        self.metrics.setdefault(f'accuracy/val', []).append(accuracy)
        self.log(f'accuracy/val', accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()


class SizeOrdinalClassifier(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    metrics: Dict[str, List[float]] = {}

    def __init__(
            self, data_path, train_data_prefix, val_data_prefix,
            in_node_nf, hidden_nf, out_node_nf, n_layers, batch_size, lr, torch_device,
            normalization=None,
            min_linker_size=None,
            linker_size2id=ZINC_TRAIN_LINKER_SIZE2ID,
            linker_id2size=ZINC_TRAIN_LINKER_ID2SIZE,
            task='ordinal',
    ):
        super(SizeOrdinalClassifier, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.min_linker_size = min_linker_size
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.linker_size2id = linker_size2id
        self.linker_id2size = linker_id2size
        self.gnn = SizeGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            n_layers=n_layers,
            device=torch_device,
            normalization=normalization,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device
            )
            self.val_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        elif stage == 'val':
            self.val_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_with_fragment_edges, shuffle=True)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_with_fragment_edges)

    def test_dataloader(self):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_with_fragment_edges)

    def forward(self, data):
        h = data['one_hot']
        x = data['positions']
        fragment_mask = data['fragment_mask']
        linker_mask = data['linker_mask']
        edge_mask = data['edge_mask']
        edges = data['edges']

        # Considering only fragments
        x = x * fragment_mask
        h = h * fragment_mask

        # Reshaping
        bs, n_nodes = x.shape[0], x.shape[1]
        fragment_mask = fragment_mask.view(bs * n_nodes, 1)
        x = x.view(bs * n_nodes, -1)
        h = h.view(bs * n_nodes, -1)

        # Prediction
        distances, _ = coord2diff(x, edges)
        distance_edge_mask = (edge_mask.bool() & (distances < 6)).long()
        output = self.gnn.forward(h, edges, distances, fragment_mask, distance_edge_mask)
        output = output.view(bs, n_nodes, -1).mean(1)
        output = sigmoid(output)

        true = self.get_true_labels(linker_mask)
        loss = self.ordinal_loss(output, true)

        return output, loss

    def ordinal_loss(self, pred, true):
        target = torch.zeros_like(pred, device=self.torch_device)
        for i, label in enumerate(true):
            target[i, 0:label + 1] = 1

        return mse_loss(pred, target, reduction='none').sum(1).mean()

    def get_true_labels(self, linker_mask):
        labels = []
        sizes = linker_mask.squeeze().sum(-1).long().detach().cpu().numpy()
        for size in sizes:
            label = self.linker_size2id.get(size)
            if label is None:
                label = self.linker_size2id[max(self.linker_id2size)]
            labels.append(label)
        labels = torch.tensor(labels, device=linker_mask.device, dtype=torch.long)
        return labels

    @staticmethod
    def prediction2label(pred):
        return torch.cumprod(pred > 0.5, dim=1).sum(dim=1) - 1

    def training_step(self, data, *args):
        _, loss = self.forward(data)
        return {'loss': loss}

    def validation_step(self, data, *args):
        _, loss = self.forward(data)
        return {'loss': loss}

    def test_step(self, data, *args):
        loss = self.forward(data)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        correct = 0
        total = 0
        for data in self.val_dataloader():
            output, _ = self.forward(data)
            pred = self.prediction2label(output)
            true = self.get_true_labels(data['linker_mask'])
            correct += (pred == true).sum()
            total += len(pred)

        accuracy = correct / total
        self.metrics.setdefault(f'accuracy/val', []).append(accuracy)
        self.log(f'accuracy/val', accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()


class SizeRegressor(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    metrics: Dict[str, List[float]] = {}

    def __init__(
            self, data_path, train_data_prefix, val_data_prefix,
            in_node_nf, hidden_nf, n_layers, batch_size, lr, torch_device,
            normalization=None, task='regression',
    ):
        super(SizeRegressor, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.gnn = SizeGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=1,
            n_layers=n_layers,
            device=torch_device,
            normalization=normalization,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.torch_device
            )
            self.val_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        elif stage == 'val':
            self.val_dataset = ZincDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.torch_device
            )
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_with_fragment_edges, shuffle=True)

    def val_dataloader(self):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_with_fragment_edges)

    def test_dataloader(self):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_with_fragment_edges)

    def forward(self, data):
        h = data['one_hot']
        x = data['positions']
        fragment_mask = data['fragment_mask']
        linker_mask = data['linker_mask']
        edge_mask = data['edge_mask']
        edges = data['edges']

        # Considering only fragments
        x = x * fragment_mask
        h = h * fragment_mask

        # Reshaping
        bs, n_nodes = x.shape[0], x.shape[1]
        fragment_mask = fragment_mask.view(bs * n_nodes, 1)
        x = x.view(bs * n_nodes, -1)
        h = h.view(bs * n_nodes, -1)

        # Prediction
        distances, _ = coord2diff(x, edges)
        distance_edge_mask = (edge_mask.bool() & (distances < 6)).long()
        output = self.gnn.forward(h, edges, distances, fragment_mask, distance_edge_mask)
        output = output.view(bs, n_nodes, -1).mean(1).squeeze()

        true = linker_mask.squeeze().sum(-1).float()
        loss = mse_loss(output, true)

        return output, loss

    def training_step(self, data, *args):
        _, loss = self.forward(data)
        return {'loss': loss}

    def validation_step(self, data, *args):
        _, loss = self.forward(data)
        return {'loss': loss}

    def test_step(self, data, *args):
        loss = self.forward(data)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        correct = 0
        total = 0
        for data in self.val_dataloader():
            output, _ = self.forward(data)
            pred = torch.round(output).long()
            true = data['linker_mask'].squeeze().sum(-1).long()
            correct += (pred == true).sum()
            total += len(pred)

        accuracy = correct / total
        self.metrics.setdefault(f'accuracy/val', []).append(accuracy)
        self.log(f'accuracy/val', accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
