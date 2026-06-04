import json
import os
import sys
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score as sk_r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from torch import nn, utils, from_numpy

from baseline import Chave


class TabularDataset(utils.data.Dataset):
    # DBH, H, WD
    MINIMUMS = [5.0, 1.2, 0.09]
    MAXIMUMS = [212.0, 70.7, 1.2]

    def __init__(self, X, y, z=None, normalize=True):
        super().__init__()
        if normalize:
            for j in range(3):
                X[:, j] /= self.MAXIMUMS[j] - self.MINIMUMS[j]
                X[:, j] -= self.MINIMUMS[j]
        self.X = from_numpy(X).to(torch.float32)
        self.y = from_numpy(y).to(torch.float32)
        self.z = from_numpy(z).to(torch.float32) if z is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.z is None:
            return self.X[i, :], self.y[i].reshape(
                1,
            )
        else:
            return (
                self.X[i, :],
                self.y[i].reshape(
                    1,
                ),
                self.z[i].reshape(
                    1,
                ),
            )


class GradNormMixin:
    def gradnorms(self):
        with torch.no_grad():
            total_norm = 0
            named_parameters = [(n, p) for n, p in self.named_parameters() if p.grad is not None and p.requires_grad]
            named_grads = dict()
            for n, p in named_parameters:
                param_norm = p.grad.detach().data.norm(2)
                named_grads[n] = param_norm.item() ** 2
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            return total_norm, named_grads

    def nb_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LinearAGBRegressor(nn.Module, GradNormMixin):

    name = "LogReg"

    epochs = 10000
    lr = 1e-2
    wd = 1e-3

    def __init__(self, **kwargs):
        self.wd = kwargs.pop("wd", 1e-5)
        super().__init__(**kwargs)
        self.regressor = nn.Linear(3, 1)

    def _init_weights(self):
        nn.init.normal_(self.regressor.bias, np.log(957.0), 1)
        nn.init.normal_(self.regressor.weight, 0, 0.1)

    def representation(self, x):
        return torch.log(x[:, :3])

    def forward(self, x):
        r = self.representation(x)
        p = self.regressor(r)
        return p.reshape(-1, 1)


class TwoLayersAGBRegressor(nn.Module, GradNormMixin):

    name = "LogReg-NN"

    epochs = 10000
    lr = 1e-3
    wd = 1e-4

    def __init__(self, **kwargs):
        self.wd = kwargs.pop("wd", 1e-5)
        super().__init__(**kwargs)
        self.representer = nn.Linear(3, 3)
        self.regressor = nn.Linear(3, 1)

    def _init_weights(self):
        nn.init.normal_(self.regressor.bias, np.log(957.0), 1)
        nn.init.normal_(self.regressor.weight, 0, 0.1)

    def representation(self, x):
        return self.representer(torch.log(x[:, :3]))

    def forward(self, x):
        r = self.representation(x)
        p = self.regressor(r)
        return p.reshape(-1, 1)


class RichRegressor(nn.Module, GradNormMixin):

    name = "COFARM"

    epochs = 15000
    lr = 1e-3
    wd = 1e-5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = 2
        self.continent_embedding = nn.Embedding(3, self.embedding_dim)
        self.forest_type_embedding = nn.Embedding(5, self.embedding_dim)
        self.representation_dim = 3 + 2 * self.embedding_dim + 4
        self.regressor = nn.Linear(self.representation_dim, 1)

    def _init_weights(self):
        nn.init.normal_(self.regressor.bias, np.log(957.0), 1)
        nn.init.normal_(self.regressor.weight, 0, 0.1)

    def representation(self, x):
        continent = x[:, 3].int()
        oldgrowth = x[:, 4].reshape(-1, 1)
        foresttype = x[:, 5].int()
        rainfall = x[:, 6].reshape(-1, 1)
        altitude = x[:, 7].reshape(-1, 1)
        drymonths = x[:, 8].reshape(-1, 1)
        dhw = torch.log(x[:, :3])
        embedded_continent = self.continent_embedding(continent)
        embedded_forest_type = self.forest_type_embedding(foresttype)
        joint_input = torch.cat(
            (
                embedded_continent,
                embedded_forest_type,
                oldgrowth,
                rainfall,
                altitude,
                drymonths,
                dhw,
            ),
            1,
        )
        return joint_input

    def forward(self, x):
        x = self.representation(x)
        x = self.regressor(x)
        return x.reshape(-1, 1)


class WideAndDeepRichRegressor(RichRegressor):

    name = "COFARM-NN"

    epochs = 20000
    lr = 1e-3
    wd = 1e-4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.internal_dim = 4
        self.deepnet = nn.Sequential(
            nn.Linear(self.representation_dim, self.internal_dim),
            nn.ReLU(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.ReLU(),
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.ReLU(),
            nn.Linear(self.internal_dim, self.internal_dim),
        )
        self.regressor = nn.Linear(self.internal_dim + 3, 1)

    def forward(self, x):
        dhw = torch.log(x[:, :3])
        x = self.representation(x)
        x = self.deepnet(x)
        x = torch.cat((x, dhw), 1)
        x = self.regressor(x)
        return x.reshape(-1, 1)


def bucketize(y, quantiles=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])):
    quantiles_values = np.quantile(y, quantiles)

    def to_agb_class(yy: float):
        for i, agb_cut in enumerate(quantiles_values):
            if yy < agb_cut:
                return i
        return i + 1

    agb_classes = np.array([to_agb_class(yy) for yy in y])
    return agb_classes


def to_xy(dff, rich_features: bool = True, return_col: str | None = None):
    if rich_features:
        dff = dff[~np.isnan(dff["OldGrowth"])]
    Xnum = dff[["DBH", "H", "WD"]].values
    continent_encoder = LabelEncoder()
    Xcontinent = continent_encoder.fit_transform(dff.Continent).reshape(-1, 1)
    if rich_features:
        old_growth = dff["OldGrowth"].fillna(0.5).values.reshape(-1, 1)
        forest_type_encoder = LabelEncoder()
        forest_type = forest_type_encoder.fit_transform(dff["ForestType"].fillna("unknown")).reshape(-1, 1)
        # scaling for happy NN optimization:
        rainfall = dff["Rainfall"].fillna(0).values.reshape(-1, 1) / 3000.0  # relative to mean annual in tropic re:JANOWIAK, 91
        altitude = dff["Altitude"].fillna(0).values.reshape(-1, 1) / 3000.0  # relative to max altitude
        drymonths = dff["DryMonths"].fillna(0).values.reshape(-1, 1) / 12.0  # per year
        X = np.hstack([Xnum, Xcontinent, old_growth, forest_type, rainfall, altitude, drymonths])
    else:
        X = np.hstack([Xnum, Xcontinent])
    y = dff.AGB.values
    col_values = dff[return_col].values if return_col else None
    return X, y, col_values


class HGBRT(HistGradientBoostingRegressor):
    name = "HGBRT"
    space = {
        "learning_rate": Real(0.001, 2, prior="log-uniform"),
        "max_leaf_nodes": Integer(64, 256),
        "max_depth": Integer(10, 30),
        "min_samples_leaf": Integer(10, 100, prior="log-uniform"),
        "l2_regularization": Real(0, 0.1),
        "max_bins": Integer(32, 255),
    }

    def __init__(
        self,
        learning_rate=0.008,
        max_leaf_nodes=156,
        max_depth=26,
        min_samples_leaf=10,
        l2_regularization=0.087,
        max_bins=208,
        max_iter=1000,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            max_iter=max_iter,
            **kwargs,
        )

    def fit(self, X, y):
        X_log = np.hstack((np.log(X[:, :3]), X[:, 3:]))
        y_log = np.log(y)
        super().fit(X_log, y_log)

    def predict(self, X):
        X_log = np.hstack((np.log(X[:, :3]), X[:, 3:]))
        y_hat_log = super().predict(X_log)
        return np.exp(y_hat_log)

    def nb_parameters(self) -> int:
        return self.max_leaf_nodes * self.max_iter


class RichChave(Chave):

    name = "ContextualChave"
    eps = 1e-8

    def __init__(self):
        self.alpha = 1.0
        self.beta0 = 1.0
        self.beta1 = 1.0
        self.beta2 = 1.0
        self.beta3 = 1.0
        self.beta4 = 1.0
        self.beta5 = 1.0
        self.beta6 = 1.0
        self.epsilon = 0.0

    @staticmethod
    def log_chave(z, X, alpha, beta0, beta1, beta2, beta3, beta4, beta5, beta6):
        return (
            alpha
            + beta0 * np.log(z)
            + beta1 * np.log(X[:, 3] + RichChave.eps)
            + beta2 * np.log(X[:, 4] + RichChave.eps)
            + beta3 * np.log(X[:, 5] + RichChave.eps)
            + beta4 * np.log(X[:, 6])
            + beta5 * np.log(X[:, 7] + RichChave.eps)
            + beta6 * np.log(X[:, 8] + RichChave.eps)
        )

    @staticmethod
    def chave_with_epsilon(z, X, alpha, beta0, beta1, beta2, beta3, beta4, beta5, beta6, epsilon):
        return (
            np.exp(alpha + epsilon)
            * (z**beta0)
            * ((X[:, 3] + RichChave.eps) ** beta1)
            * ((X[:, 4] + RichChave.eps) ** beta2)
            * ((X[:, 5] + RichChave.eps) ** beta3)
            * (X[:, 6] ** beta4)
            * ((X[:, 7] + RichChave.eps) ** beta5)
            * ((X[:, 8] + RichChave.eps) ** beta6)
        )

    def fit(self, X, y):

        def wrapper(
            X,
            alpha,
            beta0,
            beta1,
            beta2,
            beta3,
            beta4,
            beta5,
            beta6,
        ):
            z = self.z(X)
            return self.log_chave(
                z,
                X,
                alpha,
                beta0,
                beta1,
                beta2,
                beta3,
                beta4,
                beta5,
                beta6,
            )

        ly = np.log(y)
        p0 = (1, 1, 1, 1, 1, 1, 1, 1)
        popt, pcov = curve_fit(wrapper, X, ly, p0, method="lm")
        self.alpha = popt[0]
        self.beta0 = popt[1]
        self.beta1 = popt[2]
        self.beta2 = popt[3]
        self.beta3 = popt[4]
        self.beta4 = popt[5]
        self.beta5 = popt[6]
        self.beta6 = popt[7]
        self.epsilon = self.compute_epsilon(X, y)

    def compute_epsilon(self, X, y):
        z = self.z(X)
        ly = np.log(y)
        ly_hat = self.log_chave(z, X, self.alpha, self.beta0, self.beta1, self.beta2, self.beta3, self.beta4, self.beta5, self.beta6)
        residuals = ly_hat - ly
        N = len(y)
        p = 8  # alpha & betas
        sigma = np.sqrt(np.sum(residuals**2) / (N - p))
        return (sigma**2) / 2

    def predict(self, X):
        z = self.z(X)
        return self.chave_with_epsilon(
            z, X, self.alpha, self.beta0, self.beta1, self.beta2, self.beta3, self.beta4, self.beta5, self.beta6, self.epsilon
        )

    def nb_parameters(self) -> int:
        return 7


class Evaluator:

    SUPPORTED_MODELS = dict(
        (clz.name, clz) for clz in [LinearAGBRegressor, TwoLayersAGBRegressor, RichRegressor, WideAndDeepRichRegressor, HGBRT, RichChave]
    )

    def __init__(self, args):
        self.batch_size = 10000
        self._models = dict()
        for classname in args.models:
            self._models[classname] = self.SUPPORTED_MODELS[classname]
        self.result_file = args.result_file
        self.checkpoint = args.checkpoint

    @property
    def model_names(self):
        return list(self._models.keys())

    def model(self, name: str):
        return self._models[name]

    def itermodels(self):
        for k, v in self._models.items():
            yield k, v

    def run(self, X_train, y_train, X_test, y_test, args, split_id, result_data, raw_data):
        print(f"Fit shape: Xtr: {X_train.shape} ytr: {y_train.shape} Xte: {X_test.shape}")
        for model_name, model_class in self.itermodels():
            mx, y_train_hat, y_test_hat = self.run_one_model(X_test, X_train, args, model_class, model_name, y_test, y_train)
            result_data["model"].append(model_name)
            result_data["split"].append(split_id)
            for k, v in mx.items():
                result_data[k].append(v)
            raw_data[split_id][model_name]["y_train"] = y_train
            raw_data[split_id][model_name]["y_test"] = y_test
            raw_data[split_id][model_name]["y_train_hat"] = y_train_hat
            raw_data[split_id][model_name]["y_test_hat"] = y_test_hat
        return result_data

    def run_one_model(self, X_test, X_train, args, model_class, model_name, y_test, y_train):

        print(f"********** Training {model_name} **********")
        model = model_class()
        print(model)
        print(model.nb_parameters(), "parameters")

        if isinstance(model, nn.Module):
            y_test_hat, y_train_hat = self.train_pytorch_model(X_test, X_train, model, y_train)
        else:
            if args.optimize:
                model = self.optimize_hyperparams(model, X_train, y_train)
            else:
                model.fit(X_train, y_train)
            y_train_hat = model.predict(X_train)
            y_test_hat = model.predict(X_test)

        print(">>> Evaluation")
        mx = compute_metrics(y_train, y_train_hat, y_test_hat, y_test)
        return mx, y_train_hat, y_test_hat

    def optimize_hyperparams(self, model, X, y):
        print(f"Running hyperparameters optimization for {model.name}...")
        search = BayesSearchCV(
            estimator=model,
            search_spaces=model.space,
            n_iter=50,
            cv=10,
            n_jobs=1,
            verbose=0,
            random_state=42,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            search.fit(X, y)
            print(f"Best parameters: {search.best_params_}")
        return search.best_estimator_  # Already fitted

    def train_pytorch_model(self, X_test, X_train, model: nn.Module, y_train):
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, np.zeros(len(X_test))])
        is_train_idx = np.hstack([np.ones(len(X_train)), np.zeros(len(X_test))])
        train_dataset = TabularDataset(X, y, is_train_idx, normalize=False)
        train_loader = utils.data.DataLoader(
            train_dataset,
            batch_size=10**5,
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )

        model._init_weights()
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.wd)

        start = time()
        initial_loss = None
        last_loss = 1e99
        last_print_time = start

        epoch = 0
        stop = False
        patience_amount = 30
        patience = patience_amount
        force_print = False

        best_epoch = 0
        best_loss = 1e99
        os.makedirs("checkpoints", exist_ok=True)

        while not stop:
            epoch += 1
            for batch_idx, (X_b, y_b, z_b) in enumerate(train_loader):

                optimizer.zero_grad()

                is_train = torch.where(z_b > 0)[0]
                is_test = torch.where(z_b <= 0)[0]

                x_train = X_b[is_train]
                x_test = X_b[is_test]
                y_train_ = y_b[is_train]
                log_y_train = torch.log(y_train_)

                if initial_loss is None:
                    with torch.no_grad():
                        initial_loss = nn.functional.mse_loss(model(x_train), log_y_train).item()

                y_train_hat = model(x_train)
                loss = nn.functional.mse_loss(y_train_hat, log_y_train)
                loss.backward()

                max_norm_ = 199.99
                nn.utils.clip_grad_norm_(model.parameters(), max_norm_, error_if_nonfinite=True)

                optimizer.step()

            rel_loss_decrease = (last_loss - loss.item()) / last_loss
            if rel_loss_decrease < 1e-7:
                patience -= 1
                if patience < 10:
                    print(
                        "low loss decrease:",
                        rel_loss_decrease,
                        last_loss,
                        loss.item(),
                        "patience:",
                        patience,
                    )
                if patience == 0:
                    stop = True
                    if self.checkpoint:
                        print("-> reloading best model weights from epoch", best_epoch, "loss:", best_loss)
                        model = torch.load("checkpoints/" + model.name + ".pt", weights_only=False)
            else:
                patience = patience_amount
                if loss.item() < best_loss and self.checkpoint:
                    best_epoch = epoch
                    best_loss = loss.item()
                    torch.save(model, "checkpoints/" + model.name + ".pt")

            last_loss = loss.item()

            if epoch > model.epochs:
                print("max epochs reached")
                stop = True

            now = time()
            delta_from_start = timedelta(seconds=now - start)
            time_since_last_print = now - last_print_time
            if epoch <= 10 or (epoch > 1 and time_since_last_print > 4) or stop or force_print:
                last_print_time = now
                force_print = False
                with torch.no_grad():
                    log_y_train_hat = model(x_train)
                    y_train_hat = torch.exp(log_y_train_hat)
                    train_mae = torch.mean(torch.abs(y_train_ - y_train_hat))
                    mse_loss = nn.functional.mse_loss(log_y_train_hat, log_y_train).item()
                    total_norm, named_grads = model.gradnorms()
                    try:
                        train_embedding = model.representation(x_train)
                        test_embedding = model.representation(x_test)
                        embed_dist = torch.norm(
                            train_embedding.mean(dim=0) - test_embedding.mean(dim=0),
                            p=2,
                        )
                    except AttributeError:
                        embed_dist = np.nan
                print(
                    f"{delta_from_start} Epoch:{epoch:10d} ({(now - start) / epoch:2.2f} s/ep) "
                    f"| Loss:{loss.item():.5f} = MSE: {mse_loss:.5f} | align2:{embed_dist:.5f} > "
                    f"grad:{total_norm:.5f} | MAE: {train_mae:.5f}) "
                )

        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(X_test).to(torch.float32))
            y_test_hat = torch.exp(predictions).ravel().detach().numpy()
            predictions = model(torch.from_numpy(X_train).to(torch.float32))
            y_train_hat = torch.exp(predictions).ravel().detach().numpy()
        return y_test_hat, y_train_hat


def evaluate_one_data_split(df, split_id, args, result_data, raw_data):
    X, y, _ = to_xy(df, args.rich_features)
    agb_classes = bucketize(y)
    if args.cross_val == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=split_id,
            stratify=agb_classes,
        )
    elif args.cross_val == "one-site-out":
        X, y, site_values = to_xy(df, args.rich_features, "S")
        X_train = X[site_values != split_id]
        y_train = y[site_values != split_id]
        X_test = X[site_values == split_id]
        y_test = y[site_values == split_id]
        if len(y_test) == 0:
            print(f"Skipping {split_id} split because it has no sample.")
            return
    else:
        raise ValueError("Unsupported cross-validation strategy:", args.cross_val)
    print(">>> Train Size:", len(y_train), "Test Size:", len(y_test))
    print(">>> Max AGB Train:", max(y_train), "Max AGB Test:", max(y_test))

    eval_chave_baseline(X_test, X_train, y_test, y_train, split_id, result_data, raw_data)

    evaluator = Evaluator(args)
    evaluator.run(X_train, y_train, X_test, y_test, args, split_id, result_data, raw_data)


def compute_metrics(y_train, y_train_hat, y_test_hat, y_test):
    train_mae = np.abs(y_train - y_train_hat).mean()
    train_rmse = np.sqrt(((y_train - y_train_hat) ** 2).mean())
    train_r2 = sk_r2_score(y_train, y_train_hat)
    mae = np.abs(y_test - y_test_hat).mean()
    rmse = np.sqrt(((y_test - y_test_hat) ** 2).mean())
    r2 = sk_r2_score(y_test_hat, y_test)
    metrics = {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    }
    return metrics


def eval_chave_baseline(X_test, X_train, y_test, y_train, split_id, result_data, raw_data):
    chave_model = Chave()
    chave_model.fit(X_train, y_train)

    y_train_hat = chave_model.predict(X_train, y_train)
    train_mae_tobeat = np.abs(y_train - y_train_hat).mean()
    train_rmse_tobeat = np.sqrt(((y_train - y_train_hat) ** 2).mean())
    train_r2_tobeat = sk_r2_score(y_train, y_train_hat)

    y_test_hat = chave_model.predict(X_test, y_test)
    mae_tobeat = np.abs(y_test - y_test_hat).mean()
    rmse_tobeat = np.sqrt(((y_test - y_test_hat) ** 2).mean())
    r2_tobeat = sk_r2_score(y_test, y_test_hat)

    result_data["model"].append("chave")
    result_data["split"].append(split_id)
    result_data["train_mae"] += [train_mae_tobeat]
    result_data["train_rmse"] += [train_rmse_tobeat]
    result_data["train_r2"] += [train_r2_tobeat]
    result_data["test_mae"] += [mae_tobeat]
    result_data["test_rmse"] += [rmse_tobeat]
    result_data["test_r2"] += [r2_tobeat]

    raw_data[split_id]["chave"]["y_train"] = y_train
    raw_data[split_id]["chave"]["y_test"] = y_test
    raw_data[split_id]["chave"]["y_train_hat"] = y_train_hat
    raw_data[split_id]["chave"]["y_test_hat"] = y_test_hat


def write_results_summary(result_df, fd=sys.stdout, latex_fn=None):
    n = result_df.split.max() + 1 if type(result_df.split.max()) == int else len(result_df.split.unique())
    latex_preamble = r"""
    \begin{tabular}{lllr}
    \toprule
    Model & Dataset & Metric & Value (Stat. Sign.) [$\Delta$] \\
    \midrule
    """
    latex = latex_preamble
    print("*" * 80, file=fd)
    print(f"\t Train/Test Splits: {n}", file=fd)
    print("-" * 80, file=fd)
    for prefix in ("train", "test"):
        for metric in ("rmse", "mae", "r2"):
            print(file=fd)
            for model in result_df.model.unique():
                avg = result_df[result_df.model == model][f"{prefix}_{metric}"].mean()
                std = result_df[result_df.model == model][f"{prefix}_{metric}"].std()
                lb = avg - 1.96 * std / np.sqrt(n)
                ub = avg + 1.96 * std / np.sqrt(n)

                ref_avg = result_df[result_df.model == "chave"][f"{prefix}_{metric}"].mean()
                ref_std = result_df[result_df.model == "chave"][f"{prefix}_{metric}"].std()
                ref_lb = ref_avg - 1.96 * ref_std / np.sqrt(n)
                ref_ub = ref_avg + 1.96 * ref_std / np.sqrt(n)

                sig = ""
                pc = ""
                if metric == "r2":
                    if lb > ref_ub:
                        sig = "*"
                        pc = ((avg - ref_avg) / ref_avg) * 100
                        pc = f"[{pc:+.0f}%]"
                else:
                    if ub < ref_lb:
                        sig = "*"
                        pc = ((avg - ref_avg) / ref_avg) * 100
                        pc = f"[{pc:+.0f}%]"
                print(
                    f"{model:10s} {prefix:5s} {metric:4s}: {avg:.3f} +/- {1.96 * std / np.sqrt(n):.3f} {sig} {pc}",
                    file=fd,
                )
                pc = pc.replace("%", r"\%")
                latex += (
                    f"    {model} & {prefix} & {metric:4s} & {avg:.3f} " + r"$\pm$" + f"{1.96 * std / np.sqrt(n):.3f} {sig} {pc}" + r" \\ " + "\n"
                )
        print("." * 80, file=fd)
    latex += r"""
    \bottomrule
    \end{tabular}
    """
    if latex_fn is not None:
        with open(latex_fn, "w") as fd2:
            fd2.write(latex)


def report(data):
    df = pd.DataFrame(data)
    write_results_summary(df)
    df.to_csv(f"{args.result_file}.csv", index=False)
    with open(f"{args.result_file}.txt", "w") as fd:
        write_results_summary(df, fd=fd, latex_fn=f"{args.result_file}.tex")


def export_raw_data(raw_data, args):
    js = defaultdict(lambda: defaultdict(dict))
    for split, data in raw_data.items():
        for model, vectors in data.items():
            for vector_name, vector_value in vectors.items():
                js[split][model][vector_name] = [float(_) for _ in vector_value]
    json.dump(js, open(f"{args.result_file}.raw.json", "w"))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-runs", "-r", type=int, default=30)
    parser.add_argument("--test-size", "-ts", type=float, default=0.2)
    parser.add_argument("--cross-val", "-cv", choices=["one-site-out", "random"], default="random")

    parser.add_argument("--rich-features", "-ff", action="store_true", default=True)
    parser.add_argument("--no-rich-features", "-nrf", action="store_false", dest="rich_features")

    parser.add_argument("--optimize", "-op", action="store_true")

    parser.add_argument(
        "--models",
        "-m",
        default=list(Evaluator.SUPPORTED_MODELS.keys()),
        type=lambda s: s.split(","),
    )

    parser.add_argument("--checkpoint", "-cp", action="store_true", default=False)

    parser.add_argument("--result-file", "-o", default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    if args.result_file is None:
        args.result_file = "results_" + "-".join(args.models)

    with open(f"{args.result_file}.cfg", "w") as fd:
        fd.write(str(args) + "\n")

    df = pd.read_csv("chave.csv")

    result_data = defaultdict(list)
    raw_data = defaultdict(lambda: defaultdict(dict))

    if args.cross_val == "random":
        cv_runs = range(args.n_runs)
        run_info = "RUN "
    else:
        cv_runs = df.S.unique()
        run_info = "LEAVE OUT "
    for j in cv_runs:
        print("*" * 80)
        print(f"{run_info}{j}")
        print("*" * 80)
        evaluate_one_data_split(df, j, args, result_data, raw_data)
        report(result_data)
        export_raw_data(raw_data, args)
