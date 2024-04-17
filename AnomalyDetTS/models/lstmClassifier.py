#tensorboard --logdir lightning_logs
import torch
import pytorch_lightning as pl
import pandas as pd
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime

class signalsDataModule(pl.LightningDataModule):
    def __init__(self, train_signals, batch_size, test_signals = [], val_ratio = 0.2) -> None:
        super().__init__()
        self.train_signals, self.val_signals = train_test_split(train_signals, test_size = val_ratio)
        self.test_signals = test_signals
        self.batch_size = batch_size
    
    def setup(self, stage = None):
        self.train_dataset = signalDataset(self.train_signals)
        self.val_dataset = signalDataset(self.val_signals)
        self.test_dataset = signalDataset(self.test_signals)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            persistent_workers=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            persistent_workers=True
        )
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count()
        )
class signalDataset(Dataset):
    def __init__(self, signals) -> None:
        self.signals = signals
    def __len__(self):
        return len(self.signals)
    def __getitem__(self, index):
        signal, label = self.signals[index]
        return dict(
            signal = torch.Tensor(signal.to_numpy()),
            label = torch.tensor(label).long()
        )
    
class LSTMModule(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256, n_layers = 3) -> None:
        super().__init__()
        self.n_hidden = n_hidden

        self.ltsm = nn.LSTM(input_size = n_features,
                            hidden_size = n_hidden,
                            num_layers = n_layers,
                            batch_first = True,
                            dropout = 0.75)
        self.classifier = nn.Linear(n_hidden, n_classes)
    
    def forward(self, x):
        self.ltsm.flatten_parameters()
        _, (hidden, _) = self.ltsm(x)
        out = hidden[-1]
        sigm = nn.Sigmoid()
        out_ = sigm(out)
        return self.classifier(out_)


class lstmClassifier(pl.LightningModule):
    def __init__(self, n_features, n_classes) -> None:
        super().__init__()
        self.model = LSTMModule(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=2)
    
    def forward(self, signal, labels = None):
        output = self.model(signal)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def predict(self, signal):
        def resample_ts(ts):
            return ts[600:900:2]
        self.freeze()
        signal_ = resample_ts(signal)
        if not isinstance(signal, torch.Tensor):
            signal_ = torch.Tensor(signal_.to_numpy())
        _, out = self.forward(signal_.unsqueeze(dim=0))
        self.unfreeze()
        return torch.argmax(out, dim=1).item()
    
    def train_classifier(self, train_data : tuple[pd.DataFrame, bool], N_EPOCHS : int, BATCH_SIZE : int = 54, val_ratio = 0.2):
        def resample_ts(ts):
            return ts[600:900:2]
        print("INFO: Preprocessing training data!")
        train_data_m = []
        for sig, label in train_data:
            train_data_m.append((resample_ts(sig), label))
        
        data_module = signalsDataModule(train_data_m, BATCH_SIZE, val_ratio=val_ratio)
        checkpoint_callback = ModelCheckpoint(
        dirpath = "checkpoints",
        filename="best_model",
        save_top_k = 1,
        verbose=True,
        monitor="validation_loss",
        mode="min"
        )
        logger = TensorBoardLogger("lightning_logs", name="signal_prediction")
        trainer = pl.Trainer(
        logger=logger,
        num_sanity_val_steps=2,
        enable_checkpointing=True,
        callbacks = [checkpoint_callback],
        max_epochs = N_EPOCHS
        )
        trainer.fit(self, data_module)
        

    def training_step(self, batch):
        signals = batch["signal"]
        labels = batch["label"]
        loss, outputs = self(signals, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = self.accuracy(predictions, labels)

        self.log("train_loss", loss, prog_bar = True, logger = True)
        self.log("train_accuracy", step_accuracy, prog_bar = True, logger = True)
        return {"loss": loss, "accuracy" : step_accuracy}
    
    def validation_step(self, batch):
        signals = batch["signal"]
        labels = batch["label"]
        loss, outputs = self(signals, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = self.accuracy(predictions, labels)

        self.log("validation_loss", loss, prog_bar = True, logger = True)
        self.log("validation_accuracy", step_accuracy, prog_bar = True, logger = True)
        return {"loss": loss, "accuracy" : step_accuracy}
    
    def testing_step(self, batch):
        signals = batch["signal"]
        labels = batch["label"]
        loss, outputs = self(signals, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = self.accuracy(predictions, labels)

        self.log("test_loss", loss, prog_bar = True, logger = True)
        self.log("test_accuracy", step_accuracy, prog_bar = True, logger = True)
        return {"loss": loss, "accuracy" : step_accuracy}
    
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 0.0001)
    
