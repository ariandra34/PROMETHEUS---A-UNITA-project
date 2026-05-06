
!pip install pytorch-lightning

!pip install pytorch-tcn

import pandas as pd
import glob
import numpy as np
from functools import reduce
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
from pytorch_lightning import Trainer, seed_everything
from torchsummary import summary
import getpass, time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import subprocess
from sklearn.preprocessing import StandardScaler
from pytorch_tcn import TCN
import csv
from datetime import datetime

import uuid
RUN_ID = str(uuid.uuid4())

"""#Prepare data for training"""

meteo=pd.read_csv('meteo_fully_preprocessed.csv')

meteo['time'] = pd.to_datetime(meteo['time'], errors='coerce')

meteo["hour"] = meteo["time"].dt.hour

meteo["hour_sin"] = np.sin(2 * np.pi * meteo["hour"] / 24)
meteo["hour_cos"] = np.cos(2 * np.pi * meteo["hour"] / 24)

actual=pd.read_csv('production_fully_preprocessed.csv')

actual['time'] = pd.to_datetime(actual['time'], errors='coerce')
meteo['time'] = pd.to_datetime(meteo['time'], errors='coerce')

actual['time'] = actual['time'].dt.tz_localize(None)
meteo['time'] = meteo['time'].dt.tz_localize(None)

actual= actual.sort_values('time').reset_index(drop=True)
meteo = meteo.sort_values('time').reset_index(drop=True)

all_data = pd.merge_asof(left=meteo, right=actual[['time', 'pv_production']], left_on='time', right_on='time')

for lag in [1, 2, 3, 6, 12]:
    all_data[f'pv_lag_{lag}'] = all_data['pv_production'].shift(lag)

all_data = all_data.dropna()

def solar_position(time, lat, lon):
    day_of_year = time.dt.dayofyear
    hour = time.dt.hour + time.dt.minute / 60

    decl = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))

    b = 360 * (day_of_year - 81) / 364
    eot = 9.87 * np.sin(np.deg2rad(2*b)) - 7.53*np.cos(np.deg2rad(b)) - 1.5*np.sin(np.deg2rad(b))

    tc = 4 * (lon) + eot
    lst = hour + tc / 60

    hra = 15 * (lst - 12)

    lat_rad = np.deg2rad(lat)
    decl_rad = np.deg2rad(decl)
    hra_rad = np.deg2rad(hra)

    cos_zenith = (
        np.sin(lat_rad)*np.sin(decl_rad) +
        np.cos(lat_rad)*np.cos(decl_rad)*np.cos(hra_rad))

    zenith = np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))
    return zenith

LAT = 46.227379
LON = 7.364206

all_data.loc[:, "zenith"] = solar_position(all_data["time"], LAT, LON)

def extraterrestrial_irradiance(day_of_year):
    return 1367 * (1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365))

all_data = all_data.copy()

doy = all_data["time"].dt.dayofyear
I0 = extraterrestrial_irradiance(doy)

cos_zenith = np.cos(np.deg2rad(all_data["zenith"]))
cos_zenith = np.clip(cos_zenith, 1e-6, None)

all_data.loc[:, "Kt"] = all_data["irradiance"] / (I0 * cos_zenith)
all_data.loc[:, "Kt"] = all_data["Kt"].clip(0, 1.5)

def reindl_3(Kt, zenith):
    z = np.deg2rad(zenith)

    Kd = np.zeros_like(Kt)

    mask1 = Kt <= 0.3
    mask2 = (Kt > 0.3) & (Kt <= 0.78)
    mask3 = Kt > 0.78

    Kd[mask1] = 1.02 - 0.254 * Kt[mask1] + 0.0123 * np.cos(z[mask1])
    Kd[mask2] = 1.4 - 1.749 * Kt[mask2] + 0.177 * np.cos(z[mask2])
    Kd[mask3] = 0.486 * Kt[mask3] - 0.182 * np.cos(z[mask3])

    return np.clip(Kd, 0, 1)

all_data["Kd"] = reindl_3(all_data["Kt"].values, all_data["zenith"].values)

feature_cols = ['irradiance', 'temperature', 'humidity',"precipitation", "sunshine duration", "wind direction", "wind speed", "hour_sin", "hour_cos", 'pv_lag_1', 'pv_lag_2', 'pv_lag_3', 'pv_lag_6', 'pv_lag_12']

X = all_data[feature_cols]
y = all_data['pv_production']

def create_sliding_window_splits(
    X, y,
    train_size_frac=0.5,
    val_frac=0.1,
    test_frac=0.2,
    stride=None):
    n_samples = len(X)

    test_size = int(n_samples * test_frac)
    val_size = int(n_samples * val_frac)
    train_size = int(n_samples * train_size_frac)

    if stride is None:
        stride = val_size

    max_start = n_samples - test_size - train_size - val_size
    if max_start <= 0:
        raise ValueError("Not enough data for the fractions.")

    splits = []
    start = 0

    while start <= max_start:
        train_start = start
        train_end = start + train_size

        val_start = train_end
        val_end = train_end + val_size

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        splits.append((X_train, y_train, X_val, y_val))

        start += stride

    X_test = X[n_samples - test_size:]
    y_test = y[n_samples - test_size:]

    return splits, X_test, y_test

splits, X_test, y_test = create_sliding_window_splits(
    X, y,
    train_size_frac=0.5,
    val_frac=0.1,
    test_frac=0.2,
    stride=None)

print(f"Number of sliding window splits: {len(splits)}")

for i, (X_train, y_train, X_val, y_val) in enumerate(splits):
    print(f"Split {i+1}: Train={len(X_train)}, Val={len(X_val)}")

print(f"Test set size: {len(X_test)}")

scaled_splits = []

for i, (X_train, y_train, X_val, y_val) in enumerate(splits):
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1,1))

    scaled_splits.append({'X_train': X_train_scaled, 'y_train': y_train_scaled,'X_val': X_val_scaled,'y_val': y_val_scaled,'scaler_X': scaler_X,'scaler_y': scaler_y})
    print(f"Fold {i+1}: X_train_scaled shape={X_train_scaled.shape}, X_val_scaled shape={X_val_scaled.shape}")

def create_sequences(X, y, input_steps=144, output_steps=12):
    X_seq, y_seq = [], []
    n_samples = len(X)

    for i in range(n_samples - input_steps - output_steps + 1):
        X_seq.append(X[i:i+input_steps])
        y_seq.append(y[i + input_steps : i + input_steps + output_steps])

    return np.array(X_seq), np.array(y_seq)

input_steps = 24
output_steps = 12

sequential_splits = []

for i, split in enumerate(scaled_splits):
    X_train_seq, y_train_seq = create_sequences(split['X_train'], split['y_train'], input_steps, output_steps)
    X_val_seq, y_val_seq = create_sequences(split['X_val'], split['y_val'], input_steps, output_steps)

    sequential_splits.append({
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_val': X_val_seq,
        'y_val': y_val_seq,
        'scaler_X': split['scaler_X'],
        'scaler_y': split['scaler_y']})

    print(f"Fold {i+1}: X_train_seq={X_train_seq.shape}, y_train_seq={y_train_seq.shape}, "
          f"X_val_seq={X_val_seq.shape}, y_val_seq={y_val_seq.shape}")

"""#TCN in Pytorch Lightning"""

class TCN_lightning(pl.LightningModule):
    def __init__(self, input_size, output_size, num_channels=[32, 32, 32], kernel_size=3, dropout=0.2, lr=1e-3, scheduler_patience=10, scheduler_factor=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.lr = lr

        self.loss_fn = nn.MSELoss()

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []

        self.train_rmse_per_epoch = []
        self.val_rmse_per_epoch = []

        self.train_mae_per_epoch = []
        self.val_mae_per_epoch = []

        self.train_r2_per_epoch = []
        self.val_r2_per_epoch = []


    def forward(self, x):
         x = x.permute(0, 2, 1)
         y = self.tcn(x)
         y = y.permute(0,2,1)
         y = self.fc(y)
         return y[:, -1, :]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view_as(y_hat)
        loss = self.loss_fn(y_hat, y)

        rmse = self.train_rmse(y_hat.contiguous(), y.contiguous())
        mae = self.train_mae(y_hat.contiguous(), y.contiguous())
        r2 = self.train_r2(y_hat.contiguous(), y.contiguous())

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_rmse", rmse, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_epoch=True)
        self.log("train_r2", r2, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view_as(y_hat)
        loss = self.loss_fn(y_hat, y)

        rmse = self.val_rmse(y_hat.contiguous(), y.contiguous())
        mae = self.val_mae(y_hat.contiguous(), y.contiguous())
        r2 = self.val_r2(y_hat.contiguous(), y.contiguous())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_epoch=True)
        self.log("val_r2", r2, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view_as(y_hat)
        loss = self.loss_fn(y_hat, y)

        rmse = self.test_rmse(y_hat.contiguous(), y.contiguous())
        mae  = self.test_mae(y_hat.contiguous(), y.contiguous())
        r2   = self.test_r2(y_hat.contiguous(), y.contiguous())

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_rmse", rmse, on_epoch=True)
        self.log("test_mae", mae, on_epoch=True)
        self.log("test_r2", r2, on_epoch=True)

    def configure_optimizers(self):
         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
         scheduler = {
              'scheduler': ReduceLROnPlateau(
                  optimizer,
                  mode='min',
                  factor=self.hparams.scheduler_factor,
                  patience=self.hparams.scheduler_patience,
                  min_lr=1e-6
              ),
              'monitor': 'val_loss',
              'frequency': 1,
              'interval': 'epoch'
          }
         return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics

        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
            self.train_losses_per_epoch.append(train_loss)

        train_rmse = metrics.get("train_rmse")
        if train_rmse is not None:
            train_rmse = train_rmse.item() if isinstance(train_rmse, torch.Tensor) else train_rmse
            self.train_rmse_per_epoch.append(train_rmse)

        train_mae = metrics.get("train_mae")
        if train_mae is not None:
            train_mae = train_mae.item() if isinstance(train_mae, torch.Tensor) else train_mae
            self.train_mae_per_epoch.append(train_mae)

        train_r2 = metrics.get("train_r2")
        if train_r2 is not None:
            train_r2 = train_r2.item() if isinstance(train_r2, torch.Tensor) else train_r2
            self.train_r2_per_epoch.append(train_r2)

    def on_validation_epoch_end(self):
      metrics = self.trainer.callback_metrics

      val_loss = metrics.get("val_loss")
      if val_loss is not None:
          val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
          self.val_losses_per_epoch.append(val_loss)
      val_rmse = metrics.get("val_rmse")
      if val_rmse is not None:
          val_rmse = val_rmse.item() if isinstance(val_rmse, torch.Tensor) else val_rmse
          self.val_rmse_per_epoch.append(val_rmse)

      val_mae = metrics.get("val_mae")
      if val_mae is not None:
          val_mae = val_mae.item() if isinstance(val_mae, torch.Tensor) else val_mae
          self.val_mae_per_epoch.append(val_mae)

      val_r2 = metrics.get("val_r2")
      if val_r2 is not None:
          val_r2 = val_r2.item() if isinstance(val_r2, torch.Tensor) else val_r2
          self.val_r2_per_epoch.append(val_r2)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                       X_test_seq=None, y_test_seq=None, batch_size=64):

    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    val_dataset   = TimeSeriesDataset(X_val_seq, y_val_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if X_test_seq is not None and y_test_seq is not None:
        test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader

class CSVLogger:
    def __init__(self, filename="tcn_results.csv"):
        self.filename = filename

        self.fieldnames = [
            "run_id", "level", "fold", "seed",
            "seq_len", "batch_size", "input_size", "output_size",
            "num_channels", "kernel_size", "dropout", "lr", "max_epochs",
            "val_rmse", "val_mae", "val_r2",
            "rmse_mean", "rmse_std",
            "mae_mean", "mae_std",
            "r2_mean", "r2_std",
            "cv_rmse_mean", "cv_rmse_std",
            "test_rmse", "test_mae", "test_r2",
            "test_rmse_original", "test_mae_original",
            "timestamp"]

        self.file_exists = os.path.isfile(self.filename)

    def _write_row(self, row: dict):
        row["timestamp"] = datetime.now().isoformat()

        clean_row = {k: row.get(k, "") for k in self.fieldnames}

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)

            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True

            writer.writerow(clean_row)

    def log_run(self, *, level, run_id, fold=None, seed=None,
                seq_len=None, batch_size=None, input_size=None, output_size=None,
                num_channels=None, kernel_size=None, dropout=None, lr=None, max_epochs=None,
                val_rmse=None, val_mae=None, val_r2=None,
                rmse_mean=None, rmse_std=None,
                mae_mean=None, mae_std=None,
                r2_mean=None, r2_std=None,
                cv_rmse_mean=None, cv_rmse_std=None,
                test_rmse=None, test_mae=None, test_r2=None,
                test_rmse_original=None, test_mae_original=None):

        row = {
            "run_id": run_id,
            "level": level,
            "fold": fold,
            "seed": seed,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "input_size": input_size,
            "output_size": output_size,
            "num_channels": str(num_channels),
            "kernel_size": kernel_size,
            "dropout": dropout,
            "lr": lr,
            "max_epochs": max_epochs,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "test_rmse_original": test_rmse_original,
            "test_mae_original": test_mae_original}

        self._write_row(row)

"""#Train TCN"""

seeds = [42, 123, 999, 2024, 7777]
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seq_len = 24
batch_size = 32
max_epochs = 1000

input_size = 14
output_size = 12
num_channels = [8, 16, 8]
kernel_size = 3
dropout = 0.2
lr = 1e-3

train_end_idx = len(X) - len(X_test)
X_train_full = X.iloc[:train_end_idx]
y_train_full = y.iloc[:train_end_idx]

scaler_X_full = StandardScaler()
X_train_full_scaled = scaler_X_full.fit_transform(X_train_full)

scaler_y_full = StandardScaler()
y_train_full_scaled = scaler_y_full.fit_transform(y_train_full.values.reshape(-1, 1)).flatten()

X_train_full_seq, y_train_full_seq = create_sequences(
    X_train_full_scaled, y_train_full_scaled, seq_len, output_steps)

X_test_scaled = scaler_X_full.transform(X_test)
y_test_scaled = scaler_y_full.transform(y_test.values.reshape(-1, 1)).flatten()

X_test_seq, y_test_seq = create_sequences(
    X_test_scaled, y_test_scaled, seq_len, output_steps)

val_size = int(0.1 * len(X_train_full_seq))
X_train_final, X_val_final = X_train_full_seq[:-val_size], X_train_full_seq[-val_size:]
y_train_final, y_val_final = y_train_full_seq[:-val_size], y_train_full_seq[-val_size:]

train_loader_final, val_loader_final, _ = create_dataloaders(
    X_train_final, y_train_final,
    X_val_final, y_val_final,
    batch_size=batch_size)

test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

csv_logger = CSVLogger("tcn_results.csv")
RUN_ID = str(uuid.uuid4())

base_params = {
    "seq_len": seq_len,
    "batch_size": batch_size,
    "input_size": input_size,
    "output_size": output_size,
    "num_channels": str(num_channels),
    "kernel_size": kernel_size,
    "dropout": dropout,
    "lr": lr,
    "max_epochs": max_epochs}

fold_metrics = []

for fold_idx, split in enumerate(sequential_splits):
    print(f"\nFold {fold_idx+1}")

    X_train_seq = split['X_train']
    y_train_seq = split['y_train']
    X_val_seq   = split['X_val']
    y_val_seq   = split['y_val']

    train_loader, val_loader, _ = create_dataloaders(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        batch_size=batch_size)

    fold_seeds = []

    for seed in seeds:
        print(f"Seed {seed}")

        seed_everything(seed, workers=True)

        model = TCN_lightning(
            input_size, output_size,
            num_channels, kernel_size, dropout, lr)

        early_stopping = EarlyStopping(
            monitor="val_rmse",
            patience=50,
            mode="min",
            verbose=False)

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            callbacks=[early_stopping],
            log_every_n_steps=10,
            deterministic=True,
            enable_progress_bar=False)

        trainer.fit(model, train_loader, val_loader)

        val_metrics = trainer.validate(model, val_loader, verbose=False)[0]

        rmse = val_metrics["val_rmse"]
        mae  = val_metrics["val_mae"]
        r2   = val_metrics["val_r2"]

        print(f"Validation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}")

        csv_logger.log_run(
            **base_params,
            run_id=RUN_ID,
            level="seed",
            fold=fold_idx + 1,
            seed=seed,
            val_rmse=rmse,
            val_mae=mae,
            val_r2=r2)

        fold_seeds.append({
            "rmse": rmse,
            "mae": mae,
            "r2": r2})

    rmse_mean = np.mean([r["rmse"] for r in fold_seeds])
    rmse_std  = np.std([r["rmse"] for r in fold_seeds])
    mae_mean  = np.mean([r["mae"] for r in fold_seeds])
    mae_std   = np.std([r["mae"] for r in fold_seeds])
    r2_mean   = np.mean([r["r2"] for r in fold_seeds])
    r2_std    = np.std([r["r2"] for r in fold_seeds])

    print(f"\nFold {fold_idx+1} aggregate: RMSE {rmse_mean:.4f} +- {rmse_std:.4f}"
          f"MAE {mae_mean:.4f} += {mae_std:.4f}, R-squared {r2_mean:.4f} +- {r2_std:.4f}")

    csv_logger.log_run(
        **base_params,
        run_id=RUN_ID,
        level="fold",
        fold=fold_idx + 1,
        rmse_mean=rmse_mean,
        rmse_std=rmse_std,
        mae_mean=mae_mean,
        mae_std=mae_std,
        r2_mean=r2_mean,
        r2_std=r2_std)

    fold_metrics.append({
        "fold": fold_idx + 1,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "r2_mean": r2_mean,
        "r2_std": r2_std})

cv_rmse_mean = np.mean([f["rmse_mean"] for f in fold_metrics])
cv_rmse_std  = np.std([f["rmse_mean"] for f in fold_metrics])

csv_logger.log_run(
    **base_params,
    run_id=RUN_ID,
    level="cv",
    cv_rmse_mean=cv_rmse_mean,
    cv_rmse_std=cv_rmse_std)

print("\nFinal Training on Full Training Set")

final_model = TCN_lightning(
    input_size, output_size,
    num_channels, kernel_size, dropout, lr)

early_stopping_final = EarlyStopping(
    monitor="val_rmse",
    patience=50,
    mode="min",
    verbose=False)

trainer_final = Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    callbacks=[early_stopping_final],
    log_every_n_steps=10,
    deterministic=True,
    enable_progress_bar=True)

trainer_final.fit(final_model, train_loader_final, val_loader_final)

final_model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_hat = final_model(x_batch).cpu().numpy()
        test_preds.append(y_hat)
        test_targets.append(y_batch.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

test_preds_flat = test_preds.reshape(-1)
test_targets_flat = test_targets.reshape(-1)

test_preds_inv = scaler_y_full.inverse_transform(
    test_preds.reshape(-1, 1)).reshape(test_preds.shape)

test_targets_inv = scaler_y_full.inverse_transform(
    test_targets.reshape(-1, 1)).reshape(test_targets.shape)

test_rmse = np.sqrt(np.mean((test_preds - test_targets)**2))
test_mae  = mean_absolute_error(test_targets, test_preds)
test_r2   = r2_score(test_targets, test_preds)

print(f"\nTest set performance:")
print(f"RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R-squared: {test_r2:.4f}")

y_test_inv = scaler_y_full.inverse_transform(test_targets.reshape(-1, 1)).flatten()
preds_inv  = scaler_y_full.inverse_transform(test_preds.reshape(-1, 1)).flatten()

test_rmse_orig = np.sqrt(np.mean((preds_inv - y_test_inv)**2))
test_mae_orig  = mean_absolute_error(y_test_inv, preds_inv)

print(f"Test (original scale) -> RMSE: {test_rmse_orig:.2f}, MAE: {test_mae_orig:.2f}")

csv_logger.log_run(
    **base_params,
    run_id=RUN_ID,
    level="final",
    test_rmse=test_rmse,
    test_mae=test_mae,
    test_r2=test_r2,
    test_rmse_original=test_rmse_orig,
    test_mae_original=test_mae_orig
)

print("\nTraining complete. Results saved to CSV.")

df = pd.read_csv("tcn_results.csv")
print(df.columns)
print(df.head())

if hasattr(final_model, 'train_losses_per_epoch') and hasattr(final_model, 'val_losses_per_epoch'):
    train_len = len(final_model.train_losses_per_epoch)
    val_len   = len(final_model.val_losses_per_epoch)
    min_len = min(train_len, val_len)
    if train_len != val_len:
        print(f"Note: train and val loss lengths differ ({train_len} vs {val_len}). Using first {min_len} entries.")

    epochs = list(range(1, min_len + 1))
    train_loss = final_model.train_losses_per_epoch[:min_len]
    val_loss   = final_model.val_losses_per_epoch[:min_len]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training and Validation Loss', 'Training and Validation RMSE'),
        shared_xaxes=False)

    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train Loss', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Val Loss', line=dict(color='orange')), row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=1, col=1)

    if hasattr(final_model, 'train_rmse_per_epoch') and hasattr(final_model, 'val_rmse_per_epoch'):
        train_rmse = final_model.train_rmse_per_epoch[:min_len]
        val_rmse   = final_model.val_rmse_per_epoch[:min_len]

        fig.add_trace(go.Scatter(x=epochs, y=train_rmse, mode='lines', name='Train RMSE', line=dict(color='green')), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_rmse, mode='lines', name='Val RMSE', line=dict(color='red')), row=1, col=2)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        fig.update_yaxes(title_text='RMSE', row=1, col=2)
    else:
        fig.add_annotation(text="RMSE data not available", x=0.5, y=0.5, showarrow=False, row=1, col=2)

    fig.update_layout(title='Training Progress', hovermode='x unified', height=400, width=900)
    fig.show()
else:
    print("Loss history not available in final_model. Make sure the model stored the per‑epoch lists.")

y_test_inv = scaler_y_full.inverse_transform(test_targets.reshape(-1, 1)).flatten()
preds_inv = scaler_y_full.inverse_transform(test_preds.reshape(-1, 1)).flatten()

num_test_seqs = len(X_test_seq)

test_start_positions = np.arange(len(y_test))[input_steps : input_steps + num_test_seqs]

pred_positions = np.repeat(test_start_positions, output_steps) + np.tile(np.arange(output_steps), num_test_seqs)

actual_at_pred_positions = y_test.iloc[pred_positions].values

test_indices = y_test.index
timestamps_actual_line = all_data.loc[test_indices[input_steps:], 'time']
actual_line_values = y_test.iloc[input_steps:].values

timestamps_pred = all_data.loc[test_indices[pred_positions], 'time']

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=timestamps_actual_line,y=actual_line_values,mode='lines',name='Actual',line=dict(color='blue', width=1)))
fig1.add_trace(go.Scatter(x=timestamps_pred,y=preds_inv,mode='markers',name='Predicted',marker=dict(color='red', size=3, opacity=0.5)))
fig1.update_layout(title='Test Set: Actual vs Predicted PV Production',xaxis_title='Time',yaxis_title='PV Production (original scale)',hovermode='x unified')
fig1.show()

residuals = actual_at_pred_positions - preds_inv
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=timestamps_pred,y=residuals,mode='markers',name='Residuals',marker=dict(color='red', size=3, opacity=0.5)))
fig2.add_hline(y=0, line_dash="dash", line_color="black")
fig2.update_layout(title='Prediction Residuals',xaxis_title='Time',yaxis_title='Residual (Actual - Predicted)',hovermode='x unified')
fig2.show()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=actual_at_pred_positions,y=preds_inv,mode='markers',marker=dict(color='blue', size=4, opacity=0.5),name='Predictions'))

min_val = min(actual_at_pred_positions.min(), preds_inv.min())
max_val = max(actual_at_pred_positions.max(), preds_inv.max())
fig3.add_trace(go.Scatter(x=[min_val, max_val],y=[min_val, max_val],mode='lines',line=dict(color='red', dash='dash'),name='Perfect prediction'))
fig3.update_layout(title='Predicted vs Actual (test set, all forecast steps)',xaxis_title='Actual',yaxis_title='Predicted',hovermode='closest')
fig3.show()

y_test_inv = scaler_y_full.inverse_transform(test_targets.reshape(-1, 1)).flatten()
preds_inv = scaler_y_full.inverse_transform(test_preds.reshape(-1, 1)).flatten()

num_test_seqs = len(X_test_seq)
test_start_positions = np.arange(len(y_test))[input_steps : input_steps + num_test_seqs]

pred_positions = np.repeat(test_start_positions, output_steps) + np.tile(np.arange(output_steps), num_test_seqs)
actual_at_pred_positions = y_test.iloc[pred_positions].values

test_indices = y_test.index
timestamps_actual_line = all_data.loc[test_indices[input_steps:], 'time']
actual_line_values = y_test.iloc[input_steps:].values

timestamps_pred = all_data.loc[test_indices[pred_positions], 'time']

df_pred = pd.DataFrame({'time': timestamps_pred,'pred': preds_inv,'actual': actual_at_pred_positions})

df_agg = df_pred.groupby('time').agg({'pred': 'median','actual': 'first'}).reset_index()

agg_times = df_agg['time']
agg_pred = df_agg['pred']
agg_actual = df_agg['actual']

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=timestamps_actual_line,y=actual_line_values,mode='lines',name='Actual',line=dict(color='blue', width=1)))

fig1.add_trace(go.Scatter(x=agg_times,y=agg_pred,mode='lines',name='Predicted (median)',line=dict(color='red', width=1)))

fig1.update_layout(title='Test Set: Actual vs Predicted PV Production (aggregated)',xaxis_title='Time',yaxis_title='PV Production (original scale)',hovermode='x unified')

fig1.show()

residuals = agg_actual - agg_pred
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=agg_times,y=residuals,mode='lines',name='Residuals',line=dict(color='red', width=1)))

fig2.add_hline(y=0, line_dash="dash", line_color="black")
fig2.update_layout(title='Prediction Residuals (aggregated)', xaxis_title='Time',yaxis_title='Residual (Actual - Predicted)',hovermode='x unified')
fig2.show()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=agg_actual,y=agg_pred,mode='markers',marker=dict(color='blue', size=4, opacity=0.7),name='Predictions'))

min_val = min(agg_actual.min(), agg_pred.min())
max_val = max(agg_actual.max(), agg_pred.max())
fig3.add_trace(go.Scatter(x=[min_val, max_val],y=[min_val, max_val],mode='lines',line=dict(color='red', dash='dash'),name='Perfect prediction'))
fig3.update_layout(title='Predicted vs Actual (aggregated)',xaxis_title='Actual',yaxis_title='Predicted',hovermode='closest')
fig3.show()

df_pred = pd.DataFrame({
    'time': timestamps_pred,
    'pred': preds_inv,
    'actual': actual_at_pred_positions
})

df_pred = df_pred.merge(
    all_data[['time', 'Kd']],
    on='time',
    how='left')

def kd_bin(kd):
    if kd <= 0.1:
        return "clear (0-0.1)"
    elif kd <= 0.2 and kd >= 0.1:
        return "mixed (0.1-0.2)"
    elif kd <= 0.3 and kd >= 0.2:
        return "mixed (0.2-0.3)"
    elif kd <= 0.4 and kd >= 0.3:
        return "mixed (0.3-0.4)"
    elif kd <= 0.5 and kd >= 0.4:
        return "mixed (0.5-0.6)"
    elif kd <= 0.6 and kd >= 0.5:
        return "mixed (0.5-0.6)"
    elif kd <= 0.7 and kd >= 0.6:
        return "mixed (0.6-0.7)"
    elif kd <= 0.8 and kd >= 0.7:
        return "mixed (0.7-0.8)"
    elif kd <= 0.9 and kd >= 0.8:
        return "mixed (0.8-0.9)"
    elif kd <= 1.0 and kd >= 0.9:
        return "cloudy (0.9-1.0)"

df_pred["kd_bin"] = df_pred["Kd"].apply(kd_bin)

colors = {
    "clear (0-0.1)": "#2ecc71",
    "mixed (0.1-0.2)": "#a3e635",
    "mixed (0.2-0.3)": "#facc15",
    "mixed (0.3-0.4)": "#fb923c",
    "mixed (0.4-0.5)": "#f97316",
    "mixed (0.5-0.6)": "#ea580c",
    "mixed (0.6-0.7)": "#c2410c",
    "mixed (0.7-0.8)": "#7c3aed",
    "mixed (0.8-0.9)": "#4f46e5",
    "cloudy (0.9-1.0)": "#34495e"}

fig = go.Figure()

for regime, group in df_pred.groupby("kd_bin"):
    fig.add_trace(
        go.Scatter(
            x=group["actual"],
            y=group["pred"],
            mode="markers",
            name=regime,
            marker=dict(
                size=5,
                opacity=0.6,
                color=colors.get(regime, "gray"))))

min_val = min(df_pred["actual"].min(), df_pred["pred"].min())
max_val = max(df_pred["actual"].max(), df_pred["pred"].max())

fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect prediction",
        line=dict(color="red", dash="dash")))

fig.update_layout(
    title="Predicted vs Actual PV (colored by Kd regime)",
    xaxis_title="Actual PV",
    yaxis_title="Predicted PV",
    hovermode="closest")

fig.show()

mae_per_bin = {}

for bin_name, group in df_pred.groupby("kd_bin"):
    mae = mean_absolute_error(group["actual"], group["pred"])
    mae_per_bin[bin_name] = mae

print("\nMAE per Kd regime:")
for k, v in mae_per_bin.items():
    print(f"{k}: {v:.3f}")

INSTALLED_CAPACITY = 20000

for bin_name, group in df_pred.groupby("kd_bin"):
    rmse = np.sqrt(mean_squared_error(group["actual"], group["pred"]))

    nrmse = rmse / INSTALLED_CAPACITY if INSTALLED_CAPACITY > 0 else np.nan

    print(f"{bin_name}: RMSE={rmse:.2f}, nRMSE={nrmse:.2%}")

for bin_name, group in df_pred.groupby("kd_bin"):
    mae = mean_absolute_error(group["actual"], group["pred"])

    nmae = mae / INSTALLED_CAPACITY if INSTALLED_CAPACITY > 0 else np.nan

    print(f"{bin_name}: MAE={mae:.2f}, nMAE={nmae:.2%}")

colors = {
    "clear (0-0.1)": "#2ecc71",
    "mixed (0.1-0.2)": "#a3e635",
    "mixed (0.2-0.3)": "#facc15",
    "mixed (0.3-0.4)": "#fb923c",
    "mixed (0.4-0.5)": "#f97316",
    "mixed (0.5-0.6)": "#ea580c",
    "mixed (0.6-0.7)": "#c2410c",
    "mixed (0.7-0.8)": "#7c3aed",
    "mixed (0.8-0.9)": "#4f46e5",
    "cloudy (0.9-1.0)": "#34495e"}

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_pred["time"],
        y=df_pred["actual"],
        mode="lines",
        name="Actual PV",
        line=dict(color="black", width=2)))

for regime, group in df_pred.groupby("kd_bin"):
    fig.add_trace(
        go.Scatter(
            x=group["time"],
            y=group["pred"],
            mode="markers",
            name=f"Predicted - {regime}",
            marker=dict(size=4, opacity=0.6, color=colors.get(regime, "gray"))))

fig.update_layout(
    title="PV Time Series: Actual vs Predicted (colored by Kd regime)",
    xaxis_title="Time",
    yaxis_title="PV Production",
    hovermode="x unified")

fig.show()

trainer_final.fit(final_model, train_loader_final, val_loader_final)

final_model.eval()

test_preds = []
test_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_hat = final_model(x_batch).cpu().numpy()
        test_preds.append(y_hat)
        test_targets.append(y_batch.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

horizon = 12

test_start_positions = y_test.index[input_steps : input_steps + len(test_preds)]

rows = []

for i in range(len(test_preds_inv)):
    base_idx = test_start_positions[i]

    for h in range(output_steps):
        target_idx = base_idx + h

        if target_idx >= len(all_data):
            continue

        rows.append({
            "time": all_data.loc[target_idx, "time"],
            "horizon": h + 1,
            "prediction": float(test_preds_inv[i, h]),
            "actual": float(test_targets_inv[i, h]),
            "kd": all_data.loc[target_idx, "Kd"]
        })

pred_df = pd.DataFrame(rows)
pred_df.to_csv("tcn_12step_pred.csv", index=False)
print("Saved: tcn_12step_pred.csv")