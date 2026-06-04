import os
import pickle
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch

from minimal_cofarm import Evaluator, to_xy, compute_metrics


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="path to input file (CSV format)")
    parser.add_argument("--output", "-o", required=True, help="path to output file (CSV format)")
    parser.add_argument("--model", "-m", choices=list(Evaluator.SUPPORTED_MODELS.keys()), default="COFARM-NN")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        os.mkdir("model_weights")
    except FileExistsError:
        pass

    args = parse_args()

    model_weights_filename = f"model_weights/{args.model}.pt"
    model_train_stats_filename = f"model_weights/{args.model}.txt"
    if not os.path.isfile(model_weights_filename):
        print(f"could not find {args.model} model weights -> training model now")
        model_class = Evaluator.SUPPORTED_MODELS[args.model]
        model = model_class()
        X_train, y_train, _ = to_xy(pd.read_csv("chave.csv"), rich_features=True)
        print(f"loaded {len(y_train)} training samples")
        if isinstance(model, torch.nn.Module):
            y_test_hat, y_train_hat = Evaluator.train_pytorch_model(X_train, X_train, model, y_train)
            model.eval()
            torch.save(model, model_weights_filename)
            model2 = torch.load(model_weights_filename, weights_only=False)
            model2.eval()
            log_y_train_hat2 = model2.forward(torch.from_numpy(X_train).to(torch.float32))
            y_train_hat2 = torch.exp(log_y_train_hat2).ravel().detach().numpy()
            assert np.allclose(y_train_hat, y_train_hat2)
        else:
            model.fit(X_train, y_train)
            with open(model_weights_filename, "wb") as fd:
                pickle.dump(model, fd)
        print(f"training finished")

    if args.model in ("COFARM-NN", "COFARM", "LogReg", "LogReg-NN"):
        model = torch.load(model_weights_filename, weights_only=False)
    else:
        with open(model_weights_filename, "rb") as fd:
            model = pickle.load(fd)
    print(f"loaded {args.model} model weights from {model_weights_filename}")

    if not os.path.isfile(model_train_stats_filename):
        X_train, y_train, _ = to_xy(pd.read_csv("chave.csv"), rich_features=True)
        if args.model in ("COFARM-NN", "COFARM", "LogReg", "LogReg-NN"):
            with torch.no_grad():
                log_y_train_hat = model.forward(torch.from_numpy(X_train).to(torch.float32))
                y_train_hat = torch.exp(log_y_train_hat).ravel().detach().numpy()
        else:
            y_train_hat = model.predict(X_train)
        stats = compute_metrics(y_train, y_train_hat, y_train_hat, y_train)
        print(stats)
        with open(model_train_stats_filename, "w") as f:
            f.write("%s" % stats)


    input_df = pd.read_csv(args.input)
    print(f"read {len(input_df)} rows from {args.input}")
    X, _, _ = to_xy(input_df, rich_features=True)

    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(X).to(torch.float32))
            y_hat = torch.exp(predictions).ravel().detach().numpy()
    else:
        y_hat = model.predict(X)

    input_df["AGB"] = y_hat
    input_df.to_csv(args.output, index=False)
    print(f"wrote AGB predictions to {args.output}")
