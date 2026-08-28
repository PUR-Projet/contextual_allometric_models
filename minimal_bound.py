from argparse import ArgumentParser
from collections import defaultdict
from typing import Tuple, Callable, List

# from hyppo.ksample import MMD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import percentileofscore, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from baseline import Chave

# SHIFTS = {
#     "altitude": lambda x: x.Altitude < 100,
#     "drymonths": lambda x: x.DryMonths < 5,
#     "foresttype": lambda x: x.ForestType == "Dry",
#     "rainfall": lambda x: x.Rainfall > 2500,
#     "continent": lambda x: x.Continent == "Asia",
#     "oldgrowth": lambda x: x.OldGrowth == 0,
# }
# REVERSE_SHIFTS = {
#    "altitude ←": lambda x: x.Altitude >= 100,
#   "drymonths ←": lambda x: x.DryMonths >= 5,
#    "foresttype ←": lambda x: x.ForestType != "Dry",
#    "rainfall ←": lambda x: x.Rainfall <= 2500,
#    "continent ←": lambda x: x.Continent != "Asia",
#    "oldgrowth ←": lambda x: x.OldGrowth != 0,
# }
NO_SHIFTS = dict([(f"random-{i + 1}", lambda x: np.random.binomial(1, 0.5, size=len(x)) > 0) for i in range(2)])
OOD_PROPORTIONS = np.linspace(0, 1, 10)

IDENTITY_FORMATER = lambda x: x
DEFAULT_FORMATER = lambda x: f"{x:.1f}"
PRECISE_FORMATTER = lambda x: f"{x:.3f}"
SCI_FORMATER = lambda x: f"{x:.2e}"
DELTA_FORMATER = lambda x: f"{x:+.1f}"

Z_COLUMNS = ["DBH", "H", "WD"]
AGB_COLUMN = "AGB"


def mae(yy, yy_hat) -> float:
    return np.mean(np.abs(yy - yy_hat))


def chave(x, a, b) -> float:
    return a * (x[:, 1] * x[:, 2] * x[:, 0] ** 2) ** b


def hdisc(DHW_s, DHW_t, alphabeta0, ball_radius_rel: float = 0.01) -> float:
    alpha0, beta0 = alphabeta0

    def chave_diff(alphabeta):
        alpha, beta = alphabeta
        avg_s = mae(chave(DHW_s, alpha0, beta0), chave(DHW_s, alpha, beta))
        avg_t = mae(chave(DHW_t, alpha0, beta0), chave(DHW_t, alpha, beta))
        return -np.abs(avg_s - avg_t)

    alpha_delta = np.abs(alpha0 * ball_radius_rel)
    beta_delta = np.abs(beta0 * ball_radius_rel)

    d = minimize(
        chave_diff,
        (alpha0, beta0),
        bounds=((alpha0 - alpha_delta, alpha0 + alpha_delta), (beta0 - beta_delta, beta0 + beta_delta)),
        method="L-BFGS-B",
    )
    if not d.success:
        print("ERR:", d)
    return -chave_diff(d.x)
    return -d.fun  # TODO: return real function call ?


def eta(DHW_s: np.ndarray, AGB_s: np.ndarray, DHW_t: np.ndarray, AGB_t: np.ndarray, mx=mae):
    DHW_C = np.vstack([DHW_s, DHW_t])
    AGB_c = np.hstack([AGB_s, AGB_t])
    chave_c = Chave().fit(DHW_C, AGB_c)
    AGB_s_hat = chave_c.predict(DHW_s)
    AGB_t_hat = chave_c.predict(DHW_t)
    return mx(AGB_s, AGB_s_hat) + mx(AGB_t, AGB_t_hat)


def determine_shifts(dff_train: pd.DataFrame):
    shifts, reverse_shifts, references = {}, {}, {}
    # shifts["altitude-medium"] = lambda x: x["Altitude"] < np.quantile(dff_train["Altitude"], 0.5)
    # shifts["altitude-extreme"] = lambda x: x["Altitude"] < np.quantile(dff_train["Altitude"], 0.25)
    # shifts["drymonths-medium"] = lambda x: x["DryMonths"] < np.quantile(dff_train["DryMonths"], 0.5)
    # shifts["drymonths-extreme"] = lambda x: x["DryMonths"] < np.quantile(dff_train["DryMonths"], 0.25)
    # shifts["rainfall-medium"] = lambda x: x["Rainfall"] < np.quantile(dff_train["Rainfall"], 0.5)
    # shifts["rainfall-extreme"] = lambda x: x["Rainfall"] < np.quantile(dff_train["Rainfall"], 0.25)
    # shifts["foresttype-extreme"] = lambda x: x["ForestType"] == "Dry"  # 17% of data
    references = {
        "altitude-low": np.quantile(dff_train["Altitude"], 0.25),
        "altitude-high": np.quantile(dff_train["Altitude"], 0.75),
        "drymonths-few": np.quantile(dff_train["DryMonths"], 0.25),
        "drymonths-many": np.quantile(dff_train["DryMonths"], 0.85),  # Instead of 0.75 to avoid identical shift with foresttype-dry
        "rainfall-low": np.quantile(dff_train["Rainfall"], 0.25),
        "rainfall-high": np.quantile(dff_train["Rainfall"], 0.75),
        "foresttype-dry": "Dry",
        "foresttype-wet": "Wet",
        "foresttype-moist": "Moist",
        "foresttype-mangrove": "MoistMangrove",
        "continent-asia": "Asia",
        "oldgrowth-0": 0,
    }
    shifts["altitude-low"] = lambda x: x["Altitude"] < references["altitude-low"]
    shifts["altitude-high"] = lambda x: x["Altitude"] > references["altitude-high"]
    shifts["drymonths-few"] = lambda x: x["DryMonths"] < references["drymonths-few"]
    shifts["drymonths-many"] = lambda x: x["DryMonths"] > references["drymonths-many"]
    shifts["rainfall-low"] = lambda x: x["Rainfall"] < references["rainfall-low"]
    shifts["rainfall-high"] = lambda x: x["Rainfall"] > references["rainfall-high"]
    # ForestType has 4 categories: Dry, Wet, Moist, MoistMangrove
    shifts["foresttype-dry"] = lambda x: x["ForestType"] == references["foresttype-dry"]  # 17% of data
    shifts["foresttype-wet"] = lambda x: x["ForestType"] == references["foresttype-wet"]  # 13% of data
    shifts["foresttype-moist"] = lambda x: x["ForestType"] == references["foresttype-moist"]
    shifts["foresttype-mangrove"] = lambda x: x["ForestType"] == references["foresttype-mangrove"]
    # Categorical variables with 2 categories:
    shifts["continent-asia"] = lambda x: x["Continent"] == references["continent-asia"]  # 28% of data
    shifts["oldgrowth-0"] = lambda x: x["OldGrowth"] == references["oldgrowth-0"]  # 14% of data
    # Reverse
    # reverse_shifts["altitude-medium"] = lambda x: x["Altitude"] >= np.quantile(dff_train["Altitude"], 0.5)
    # reverse_shifts["altitude-extreme"] = lambda x: x["Altitude"] > np.quantile(dff_train["Altitude"], 0.75)
    # reverse_shifts["drymonths-medium"] = lambda x: x["DryMonths"] >= np.quantile(dff_train["DryMonths"], 0.5)
    # reverse_shifts["drymonths-extreme"] = lambda x: x["DryMonths"] > np.quantile(dff_train["DryMonths"], 0.75)
    # reverse_shifts["rainfall-medium"] = lambda x: x["Rainfall"] >= np.quantile(dff_train["Rainfall"], 0.5)
    # reverse_shifts["rainfall-extreme"] = lambda x: x["Rainfall"] > np.quantile(dff_train["Rainfall"], 0.75)
    # reverse_shifts["foresttype-extreme"] = lambda x: x["ForestType"] == "Wet"
    reverse_shifts["altitude-low"] = lambda x: x["Altitude"] >= references["altitude-low"]
    reverse_shifts["altitude-high"] = lambda x: x["Altitude"] <= references["altitude-high"]
    reverse_shifts["drymonths-few"] = lambda x: x["DryMonths"] >= references["drymonths-few"]
    reverse_shifts["drymonths-many"] = lambda x: x["DryMonths"] <= references["drymonths-many"]
    reverse_shifts["rainfall-low"] = lambda x: x["Rainfall"] >= references["rainfall-low"]
    reverse_shifts["rainfall-high"] = lambda x: x["Rainfall"] <= references["rainfall-high"]
    reverse_shifts["foresttype-dry"] = lambda x: x["ForestType"] != references["foresttype-dry"]
    reverse_shifts["foresttype-wet"] = lambda x: x["ForestType"] != references["foresttype-wet"]
    reverse_shifts["foresttype-moist"] = lambda x: x["ForestType"] != references["foresttype-moist"]
    reverse_shifts["foresttype-mangrove"] = lambda x: x["ForestType"] != references["foresttype-mangrove"]
    reverse_shifts["continent-asia"] = lambda x: x["Continent"] != references["continent-asia"]
    reverse_shifts["oldgrowth-0"] = lambda x: x["OldGrowth"] != references["oldgrowth-0"]
    return shifts, reverse_shifts, references


def split_data(
    dff: pd.DataFrame, shift_condition: Callable[[pd.DataFrame], bool], reverse_shift_condition: Callable[[pd.DataFrame], bool] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df1 = dff[shift_condition(dff)]
    df2 = dff[reverse_shift_condition(dff)] if reverse_shift_condition is not None else dff[~shift_condition(dff)]
    return df1[Z_COLUMNS].values, df1[AGB_COLUMN].values, df2[Z_COLUMNS].values, df2[AGB_COLUMN].values


def extract_features(mae_s, z_s, z_t, z_t_std, hds, n_trains, train_agbs):
    z_s = np.asarray(z_s)
    z_t = np.asarray(z_t)
    z_t_std = np.asarray(z_t_std)
    hds = np.asarray(hds)
    mae_s = np.asarray(mae_s)
    n_trains = np.asarray(n_trains)
    train_agbs = np.asarray(train_agbs)
    reldif = (z_t - z_s) / z_s

    def sigmoid(x):
        return 1 / (1 + np.exp(-(x - 1)))

    Xp = np.hstack(
        [
            # 1 / np.sqrt(n_trains.reshape(-1, 1)),
            # (mae_s / 300).reshape(-1, 1),
            # (train_agbs / 1200).reshape(-1, 1),
            (hds / 10000).reshape(-1, 1),
            # z_s.reshape(-1,1),
            # z_t.reshape(-1,1),
            # ((z_t - z_s) / np.abs(z_s)).reshape(-1, 1),
            (np.sign(z_t - z_s)).reshape(-1, 1),
            # z_t_std.reshape(-1, 1),
            # (hds * sigmoid(reldif)).reshape(-1,1),
        ]
    )

    poly = PolynomialFeatures(2, interaction_only=False, include_bias=False)
    Xp = poly.fit_transform(Xp)
    return Xp


def test_shift_significance(test_DHW0, test_DHWa):
    _, pvalue = MMD().test(test_DHW0, test_DHWa)  # Maximal Mean Discrepency: kernel-based manova without gaussian hypothesis
    return round(pvalue, 3)


def compute_shift_dist(z_source, z_target):
    dist = wasserstein_distance(z_source, z_target)
    return round(dist, 1)


def test_shift_learnability(X1, X2, n_bootstraps: int = 100):
    train = np.concatenate([X1, X2], axis=0)
    train_is_source = np.concatenate([np.ones(X1.shape[0]), np.zeros(X2.shape[0])])
    # Dummy model for null distribution
    null_dist_ll = []
    for j in range(n_bootstraps):
        p = len(X1) / len(train)
        y_rand = np.random.binomial(1, p, len(train))
        model = HistGradientBoostingClassifier()
        model.fit(train, y_rand)
        ll_dummy = log_loss(y_rand, model.predict_proba(train)[:, 1])
        null_dist_ll += [ll_dummy]
    # RF model
    rf_model = HistGradientBoostingClassifier()
    rf_model.fit(train, train_is_source)
    y_hat = rf_model.predict_proba(train)[:, 1]
    ll = log_loss(train_is_source, y_hat)
    print("learnability:", np.quantile(null_dist_ll, [0.025, 0.975]), "vs", ll)
    return percentileofscore(null_dist_ll, ll, kind="weak")


def train_delta_mae_predictor(dff_train: pd.DataFrame, shifts: dict, reverse_shifts: dict):
    print(">>> gathering training data")
    delta_maes, hds, mae_sources, n_trains, train_agbs, z_sources, z_targets, z_targets_std = build_train_dataset(
        dff_train, shifts, reverse_shifts, n_shift_samples=2000, n_random_samples=1000
    )
    print(">>> training predictor")
    Xp = extract_features(mae_sources, z_sources, z_targets, z_targets_std, hds, n_trains, train_agbs)
    yp = np.asarray(delta_maes)
    print("train data:", Xp.shape, yp.shape)
    # sign = np.sign(yp)
    # c = HistGradientBoostingClassifier().fit(Xp, sign)
    # print("sign acc:", c.score(Xp, sign))
    m = Pipeline([("scaler", StandardScaler()), ("linreg", RidgeCV())])
    m.fit(Xp, yp)
    # m = MLPRegressor(hidden_layer_sizes=(3, 3), max_iter=20000).fit(Xp, yp)
    # m = HistGradientBoostingRegressor(max_depth=4, max_iter=10, min_samples_leaf=30, monotonic_cst=(1, 0, 0, 1, 0, 0)).fit(Xp, yp)
    try:
        print(m.steps[0][0], m.steps[0][1].mean_, m.steps[0][1].var_)
        print(m.steps[1][0], m.steps[1][1].coef_, m.steps[1][1].intercept_)
    except AttributeError:
        pass
    # p_delta = m.predict(Xp)
    # p_sign_delta = c.predict(Xp)
    # r2 = r2_score(yp, p_delta * p_sign_delta)
    r2 = r2_score(yp, m.predict(Xp))
    sign_acc = accuracy_score(np.sign(yp), np.sign(m.predict(Xp)))
    print("train sign Accuracy:", sign_acc)
    return m, r2


def build_train_dataset(dff_train, shifts, reverse_shifts, n_random_samples=1000, n_shift_samples=30):
    mae_sources = []
    mae_targets = []
    z_sources = []
    z_targets = []
    z_targets_std = []
    hds = []
    delta_maes = []
    n_trains = []
    train_agbs = []
    for _ in range(n_random_samples):
        p = np.random.uniform(0.05, 0.95)
        DHW1, AGB1, DHW2, AGB2 = split_data(dff_train, shift_condition=lambda _: np.random.binomial(1, p, size=len(dff_train)) > 0)
        gather_shift_data(
            AGB1, AGB2, DHW1, DHW2, delta_maes, hds, mae_sources, mae_targets, z_sources, z_targets, z_targets_std, n_trains, train_agbs
        )
        gather_shift_data(
            AGB2, AGB1, DHW2, DHW1, delta_maes, hds, mae_sources, mae_targets, z_sources, z_targets, z_targets_std, n_trains, train_agbs
        )
    for _ in range(n_shift_samples):
        rand_idx = np.random.randint(0, len(dff_train), size=len(dff_train))
        dff_train_reshuffled = dff_train.iloc[rand_idx]
        for shiftname, shift_condition in shifts.items():
            DHW1, AGB1, DHW2, AGB2 = split_data(
                dff_train_reshuffled, shift_condition=shift_condition, reverse_shift_condition=reverse_shifts[shiftname]
            )
            gather_shift_data(
                AGB1, AGB2, DHW1, DHW2, delta_maes, hds, mae_sources, mae_targets, z_sources, z_targets, z_targets_std, n_trains, train_agbs
            )
            gather_shift_data(
                AGB2, AGB1, DHW2, DHW1, delta_maes, hds, mae_sources, mae_targets, z_sources, z_targets, z_targets_std, n_trains, train_agbs
            )
            partial_dff_train_reshuffled_src = dff_train_reshuffled[shift_condition(dff_train_reshuffled)]
            partial_dff_train_reshuffled_tgt = dff_train_reshuffled[reverse_shifts[shiftname](dff_train_reshuffled)]

    return delta_maes, hds, mae_sources, n_trains, train_agbs, z_sources, z_targets, z_targets_std


def gather_shift_data(
    AGB1: np.ndarray,
    AGB2: np.ndarray,
    DHW1: np.ndarray,
    DHW2: np.ndarray,
    delta_maes: List[float],
    hds: List[float],
    mae_sources: List[float],
    mae_targets: List[float],
    z_sources: List[float],
    z_targets: List[float],
    z_targets_std: List[float],
    n_trains: List[float],
    train_agbs: List[float],
):
    c0 = Chave()
    c0.fit(DHW1, AGB1)
    train_alpha0, train_beta0 = c0.alpha + c0.epsilon, c0.beta
    mae_sources += [mae(AGB1, c0.predict(DHW1))]
    mae_targets += [mae(AGB2, c0.predict(DHW2))]
    delta_maes += [mae_targets[-1] - mae_sources[-1]]
    z_sources += [np.mean(DHW1[:, 0] ** 2 * DHW1[:, 1] * DHW1[:, 2])]
    z_targets += [np.mean(DHW2[:, 0] ** 2 * DHW2[:, 1] * DHW2[:, 2])]
    z_targets_std += [np.std(DHW2[:, 0] ** 2 * DHW2[:, 1] * DHW2[:, 2])]
    hds += [hdisc(DHW1, DHW2, (train_alpha0, train_beta0))]
    n_trains += [len(AGB1)]
    train_agbs += [np.mean(AGB1)]


def evaluate_delta_mae_predictor(predictor, dff_train: pd.DataFrame, dff_test: pd.DataFrame, shifts: dict, reverse_shifts: dict, args):
    raw_result_data, dist_data = defaultdict(list), defaultdict(list)
    formatters = []
    for i, (shiftname, shift_condition) in enumerate((shifts | NO_SHIFTS).items()):
        reverse_shift_condition = reverse_shifts.get(shiftname, None)
        evaluate_one_shift(
            args,
            dff_train,
            dff_test,
            shifts,
            formatters,
            i,
            predictor,
            raw_result_data,
            shift_condition,
            reverse_shift_condition,
            shiftname,
            dist_data,
        )

    test_r2 = r2_score(raw_result_data[r"$\Delta_{s,t} MAE$"], raw_result_data[r"$\hat{\Delta}_{s,t} MAE$"])

    return raw_result_data, formatters, test_r2, dist_data


def evaluate_one_shift(
    args, dff_train, dff_test, shifts, formatters, i, predictor, raw_result_data, shift_condition, reverse_shift_condition, shiftname, dist_data
):
    print(">>> evaluating delta mae predictor on", shiftname)
    seed = i % len(shifts)

    # DHW0, AGB0, _ = sample(dff_test, ood_proportion=0, n=len(dff_test) * 1000, shift_condition=shift_condition, seed=seed)
    DHW0, AGB0, DHW1, AGB1 = split_data(dff_test, shift_condition, reverse_shift_condition)

    eta_, hd0, mae_source, mae_target, predicted_delta_mae, z_source, z_target, dist = evaluate_shift_prediction(AGB0, AGB1, DHW0, DHW1, predictor)
    # dist, pval = test_shift_significance(DHW0, DHW1)
    # pval = test_shift_learnability(DHW0, DHW1)  # takes long to compute
    # pval = 1.0 if "random" in shiftname else 0.0
    dist_data["Shift"] += [shiftname]
    dist_data["Wass. Dist."] += [dist]
    dist_data[r"$\mathcal{H}Disc$"] += [hd0]
    add_metrics_to_result_data(
        AGB0,
        AGB1,
        eta_,
        formatters,
        hd0,
        mae_source,
        mae_target,
        predicted_delta_mae,
        raw_result_data,
        shiftname + "→",
        z_source,
        z_target,
    )
    eta_, hd1, mae_source, mae_target, predicted_delta_mae, z_source, z_target, _ = evaluate_shift_prediction(AGB1, AGB0, DHW1, DHW0, predictor)
    add_metrics_to_result_data(
        AGB1,
        AGB0,
        eta_,
        formatters,
        hd1,
        mae_source,
        mae_target,
        predicted_delta_mae,
        raw_result_data,
        shiftname + "←",
        z_source,
        z_target,
    )
    simulate_progressive_shift(
        AGB0, DHW0, args, dff_train, dff_test, mae_source, predictor, seed, shift_condition, reverse_shift_condition, shiftname + "→", z_source
    )
    simulate_progressive_shift(
        AGB1, DHW1, args, dff_train, dff_test, mae_source, predictor, seed, reverse_shift_condition, shift_condition, shiftname + "←", z_target
    )


def simulate_progressive_shift(
    AGB_src, DHW_src, args, dff_train, dff_test, mae_source, predictor, seed, shift_condition, reverse_shift_condition, shiftname, z_source
):
    hdiscs = []
    maes_target = []
    maes_source = []
    predicted_delta_maes = []
    zs_target = []
    zs_source = []
    if shift_condition is None:
        shift_condition = lambda x: ~reverse_shift_condition(x)
    dff_test_src = dff_test[shift_condition(dff_test)]
    if reverse_shift_condition is None:
        reverse_shift_condition = lambda x: ~shift_condition(x)
    dff_test_tgt = dff_test[reverse_shift_condition(dff_test)]

    dff_train_src = dff_train[shift_condition(dff_train)]
    # dff_train_tgt = dff_train[reverse_shift_condition(dff_train)]
    z_train_src = dff_train_src[Z_COLUMNS].values
    agb_train_src = dff_train_src["AGB"].values
    chave_model_on_train_src = Chave().fit(z_train_src, agb_train_src)

    for j, prop in enumerate(OOD_PROPORTIONS):
        n = len(dff_test)  # * 100
        n_src = int(n * (1 - prop))
        n_tgt = int(n * prop)
        dff_test_mixup_src = dff_test_src.sample(n=n_src, replace=True, random_state=seed)
        dff_test_mixup_tgt = dff_test_tgt.sample(n=n_tgt, replace=True, random_state=seed + 1)
        dff_test_mixup = pd.concat([dff_test_mixup_src, dff_test_mixup_tgt])
        dff_test_mixup = dff_test_mixup.sample(n=len(dff_test_mixup), replace=False, random_state=seed + 3)
        DHWa, AGBa = dff_test_mixup[Z_COLUMNS].values, dff_test_mixup["AGB"].values
        mae_target = mae(AGBa, chave_model_on_train_src.predict(DHWa))
        maes_target += [mae_target]
        maes_source += [mae_source]
        hd = hdisc(DHW_src, DHWa, (chave_model_on_train_src.alpha + chave_model_on_train_src.epsilon, chave_model_on_train_src.beta))
        hdiscs += [hd]
        z = DHWa[:, 0] ** 2 * DHWa[:, 1] * DHWa[:, 2]
        z_target = np.mean(z)
        z_target_std = np.std(z)
        zs_target += [z_target]
        zs_source += [z_source]
        Xp = extract_features([mae_source], [z_source], [z_target], [z_target_std], [hd], [len(AGB_src)], [np.mean(AGB_src)]).reshape(1, -1)
        predicted_delta_mae = predictor.predict(Xp)[0]
        predicted_delta_maes += [predicted_delta_mae]
    maes_source = np.array(maes_source)
    maes_target = np.array(maes_target)
    plot_shift_prediction(args, hdiscs, maes_source, maes_target, predicted_delta_maes, shiftname)


def add_metrics_to_result_data(
    AGB0,
    AGB1,
    eta_,
    formatters,
    hd,
    mae_source,
    mae_target,
    predicted_delta_mae,
    raw_result_data,
    shiftname,
    z_source,
    z_target,
):
    raw_result_data["Shift"] += [shiftname]
    formatters += [IDENTITY_FORMATER]
    raw_result_data[r"$\mathrm{E}[AGB_s]$"] += [np.mean(AGB0)]
    formatters += [DEFAULT_FORMATER]
    raw_result_data[r"$\mathrm{E}[AGB_t]$"] += [np.mean(AGB1)]
    formatters += [DEFAULT_FORMATER]
    raw_result_data[r"$\Delta Z$"] += [z_target - z_source]
    formatters += [DELTA_FORMATER]
    # raw_result_data[r"$\Delta NLLH$ (\%)"] += [100 * (ll_dummy - ll) / ll_dummy]
    # formatters += [PRECISE_FORMATTER]
    raw_result_data[r"$\mathcal{H}Disc$"] += [hd]
    formatters += [DEFAULT_FORMATER]
    raw_result_data[r"$\eta$"] += [eta_]
    formatters += [DEFAULT_FORMATER]
    raw_result_data[r"$MAE^{c}_s$"] += [mae_source]
    formatters += [DEFAULT_FORMATER]
    raw_result_data[r"$MAE^{c}_t$"] += [mae_target]
    formatters += [DEFAULT_FORMATER]
    raw_result_data[r"$\Delta_{s,t} MAE$"] += [mae_target - mae_source]
    formatters += [DELTA_FORMATER]
    raw_result_data[r"$\hat{\Delta}_{s,t} MAE$"] += [predicted_delta_mae]
    formatters += [DELTA_FORMATER]
    recovery = 100 * (predicted_delta_mae / (mae_target - mae_source))
    # raw_result_data[r"$\hat{\Delta}_{s,t} / \Delta_{s,t}$ (\%)"] += [recovery]
    # formatters += [DEFAULT_FORMATER]
    # sign_accuracy = np.mean(np.sign(predicted_delta_mae) == np.sign(mae_target - mae_source))
    # raw_result_data[r"Acc. $sign(\hat{\Delta}_{s,t})$"] += [sign_accuracy]
    # formatters += [PRECISE_FORMATTER]
    # raw_result_data["n_s/n_t"] += [dn]
    return formatters


def evaluate_shift_prediction(AGB0, AGB1, DHW0, DHW1, predictor):
    z_source = DHW0[:, 0] ** 2 * DHW0[:, 1] * DHW0[:, 2]
    z_target = DHW1[:, 0] ** 2 * DHW1[:, 1] * DHW1[:, 2]
    dist = compute_shift_dist(z_source, z_target)
    z_source_mean = np.mean(z_source)
    z_target_mean = np.mean(z_target)
    z_target_std = np.std(z_target)
    c0 = Chave()
    c0.fit(DHW0, AGB0)
    alpha0, beta0 = c0.alpha + c0.epsilon, c0.beta
    hd = hdisc(DHW0, DHW1, (alpha0, beta0))
    # print(alpha0, beta0, "-->", hd)
    eta_ = eta(DHW0, AGB0, DHW1, AGB1)

    preds_target = c0.predict(DHW1)
    mae_source = mae(AGB0, c0.predict(DHW0))
    mae_target = mae(AGB1, preds_target)

    Xp = extract_features([mae_source], [z_source_mean], [z_target_mean], [z_target_std], [hd], [len(AGB0)], [np.mean(AGB0)]).reshape(1, -1)
    # predicted_delta_sign = predictor[1].predict(Xp)
    # predicted_delta_magn = predictor[0].predict(Xp)
    # predicted_delta_mae = (predicted_delta_sign * predicted_delta_magn)[0]
    predicted_delta_mae = predictor.predict(Xp)[0]

    return eta_, hd, mae_source, mae_target, predicted_delta_mae, z_source_mean, z_target_mean, dist


def plot_shift_prediction(args, hdiscs, maes_source, maes_target, predicted_delta_maes, shiftname: str):
    fig, ax = plt.subplots()
    ax.set_title(shiftname)
    ax.plot(OOD_PROPORTIONS, hdiscs[-len(OOD_PROPORTIONS) :], label="$\\mathcal{H}-Disc$", alpha=0.5, ls=":")
    ax2 = ax.twinx()
    ax2.plot(OOD_PROPORTIONS, maes_target[-len(OOD_PROPORTIONS) :], color="orange", label=f"True MAE")
    # ax2.plot(OOD_PROPORTIONS, etas, color='red', label="$\\eta$", ls='--')
    ax2.plot(OOD_PROPORTIONS, (maes_source + predicted_delta_maes)[-len(OOD_PROPORTIONS) :], color="magenta", label=f"Predicted MAE")
    # ax2.plot(OOD_PROPORTIONS, err_preds + err0 + etas, color='chartreuse', label="MAE upper bound", ls="--")
    ax.grid()
    ax.set_ylim(0, 8000)
    ax.set_xlabel("$\\alpha$: O.O.D. proportion")
    ax.set_ylabel("$\\mathcal{H}-Disc$")
    ax2.set_ylabel("$MAE$ (kg)")
    ax2.set_ylim(0, 600)
    ax.legend(loc=(0, 0.90))
    ax2.legend(loc=(0, 0.65))
    fig.tight_layout()
    fig.savefig(f"{args.result_file}_{shiftname}.png")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test-size", "-ts", type=float, default=0.5)
    parser.add_argument("--result-file", "-o", default=None)
    args = parser.parse_args()
    return args


def run_evaluation(dff_randomized):
    n_test = int(len(df) * args.test_size)
    df_test = dff_randomized[:n_test]
    df_train = dff_randomized[n_test:]

    shifts, reverse_shifts, references = determine_shifts(df_train)

    predictor, train_r2 = train_delta_mae_predictor(df_train, shifts, reverse_shifts)
    print(predictor)
    print("train R2:", train_r2)
    with open(f"{args.result_file}.model", "w") as fd:
        print(predictor, file=fd)
        try:
            print(predictor.steps[0][0], predictor.steps[0][1].mean_, predictor.steps[0][1].var_, file=fd)
            print(predictor.steps[1][0], predictor.steps[1][1].coef_, predictor.steps[1][1].intercept_, file=fd)
        except AttributeError:
            pass
        print("model train R2:", train_r2, file=fd)

    result_data, formatters, test_r2, dist_data = evaluate_delta_mae_predictor(predictor, df_train, df_test, shifts, reverse_shifts, args)
    print("test R2:", test_r2)
    result_data = pd.DataFrame(result_data)
    # print("test sign Accuracy:", np.mean(result_data[r"$\hat{\Delta}_{s,t} / \Delta_{s,t}$ (\%)"] > 0))
    result_data.sort_values("Shift", inplace=True)
    result_data.to_csv(f"{args.result_file}.csv", index=False)
    result_data.to_latex(f"{args.result_file}.tex", index=False, formatters=formatters[: result_data.shape[1]])

    dist_data = pd.DataFrame(dist_data)
    dist_data.sort_values("Shift", inplace=True)
    dist_data["Threshold"] = dist_data["Shift"].apply(lambda x: references[x] if x in references else "-")
    dist_data["Shift"] = dist_data["Shift"]
    dist_data = dist_data.reindex(["Shift", "Threshold", "Wass. Dist.", r"$\mathcal{H}Disc$"], axis=1)
    dist_data.to_latex(
        f"{args.result_file}_dist.tex", index=False, formatters=[IDENTITY_FORMATER, IDENTITY_FORMATER, IDENTITY_FORMATER, DEFAULT_FORMATER]
    )

    plot_global_result(result_data, test_r2)


def plot_global_result(result_data: pd.DataFrame, test_r2):
    dff = result_data.copy()
    dff["FullShiftName"] = dff["Shift"]
    dff["Shift"] = dff["Shift"].apply(lambda x: x.split("-")[0].replace("1", "").replace("2", "").replace("←", "").replace("→", ""))

    lm = LinearRegression()
    x = dff[r"$\Delta_{s,t} MAE$"].values.reshape(-1, 1)
    y = dff[r"$\hat{\Delta}_{s,t} MAE$"].values
    lm.fit(x.reshape(-1, 1), y)

    # for i, (color, color_name) in enumerate(zip(colors, TABLEAU_COLORS)):
    colors = ["red", "green", "blue", "orange", "cyan", "magenta", "purple", "chartreuse", "brown"]
    markers = ["o", ">", "<", "^", "v"]
    shift_colors = dict((s, colors[i]) for i, s in enumerate(dff["Shift"].unique()))

    plt.figure(figsize=(6, 6))
    # plt.title(f"Actual vs Predicted Additional Error ($R^2 = {test_r2:.3f}$)")
    for shiftname in dff.Shift.unique():
        first_shift = True
        selector = dff["Shift"] == shiftname
        selected_dff = dff[selector]
        for j, sn in enumerate(selected_dff.FullShiftName.unique()):
            fine_selected_dff = selected_dff[selected_dff.FullShiftName == sn]
            # test_r2_real_shifts = r2_score(dff[selector][r"$\Delta_{s,t} MAE$"], dff[selector][r"$\hat{\Delta}_{s,t} MAE$"])
            plt.scatter(
                fine_selected_dff[r"$\Delta_{s,t} MAE$"],
                fine_selected_dff[r"$\hat{\Delta}_{s,t} MAE$"],
                label=f"{shiftname}" if first_shift else None,
                color=shift_colors[shiftname],
                marker=markers[j % len(markers)],
            )
            first_shift = False

    l, h = dff[r"$\Delta_{s,t} MAE$"].min(), dff[r"$\Delta_{s,t} MAE$"].max()
    l_hat, h_hat = dff[r"$\hat{\Delta}_{s,t} MAE$"].min(), dff[r"$\hat{\Delta}_{s,t} MAE$"].max()
    ll = min(l, l_hat) - 10
    hh = max(h, h_hat) + 10
    plt.plot((ll, hh), (ll, hh), "k--", alpha=0.5)
    plt.xlim(ll, hh)
    plt.ylim(ll, hh)

    y_ll = lm.predict([[ll]])
    y_hh = lm.predict([[hh]])
    plt.plot((ll, hh), (y_ll, y_hh), color="blue", linestyle=":", alpha=0.5, label=f"$R^2 = {test_r2:.3f}$")

    plt.fill_between((0, hh), (ll, ll), (0, 0), alpha=0.2, color="r")
    plt.xlabel(r"$\Delta_{s,t} MAE$")
    plt.ylabel(r"$\hat{\Delta}_{s,t} MAE$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.result_file}.png")


def test_hdisc(df_randomized):
    for _ in range(30):
        df_randomized["pop"] = np.random.binomial(1, 0.15, size=len(df_randomized))
        alpha0 = np.random.normal(0.0673, 0.1)
        beta0 = np.random.normal(0.976, 0.1)
        DHW1 = df_randomized[df_randomized["pop"] > 0][Z_COLUMNS].values
        DHW2 = df_randomized[df_randomized["pop"] == 0][Z_COLUMNS].values
        hd1 = hdisc(DHW1, DHW2, (alpha0, beta0), ball_radius_rel=0.5)
        hd2 = hdisc(DHW2, DHW1, (alpha0, beta0), ball_radius_rel=0.5)
        print(len(DHW1), len(DHW2), alpha0, beta0, hd1, hd2)


if __name__ == "__main__":

    args = parse_args()

    if args.result_file is None:
        args.result_file = "bound_results"

    with open(f"{args.result_file}.cfg", "w") as fd:
        fd.write(str(args) + "\n")

    df = pd.read_csv("chave.csv")
    df = df[df.ForestType > ""]
    df_randomized = df.sample(n=len(df), replace=False, random_state=42)

    # test_hdisc(df_randomized)

    run_evaluation(df_randomized)
