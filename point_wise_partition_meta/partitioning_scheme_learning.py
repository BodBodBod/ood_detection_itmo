from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as sk_metrics
from matplotlib import pyplot as plt
import seaborn as sns

from point_wise_partition_meta.partition_ood import (
    DEFAULT_META_FEATURES,
    DecisionTreeScheme,
    EnsemblePartitionOOD,
    KMeansScheme,
    QuantileBinningScheme,
)


@dataclass
class SingleSchemeConfig:
    scheme_type: str
    n_bins: int = 10
    n_kmeans_clusters: int = 10
    n_tree_partitions: int = 5
    tree_max_depth: int = 3
    random_state: int = 42
    meta_features: list | None = None
    use_raw_features: bool = True


class SingleSchemePartitionOOD(EnsemblePartitionOOD):
    """
    Partition-based representation that uses only one scheme family:
    quantile, kmeans, or tree.

    Cell statistics and meta-feature extraction are inherited from
    ``EnsemblePartitionOOD``, so a 36-feature taxonomy is computed in full.
    """

    def __init__(self, config: SingleSchemeConfig):
        super().__init__(
            n_bins=config.n_bins,
            n_kmeans_clusters=config.n_kmeans_clusters,
            n_tree_partitions=config.n_tree_partitions,
            tree_max_depth=config.tree_max_depth,
            random_state=config.random_state,
            meta_features=config.meta_features,
        )
        self.config = config
        self.scheme_type = config.scheme_type

    def _build_schemes(self, n_features: int):
        cfg = self.config
        rng = np.random.RandomState(cfg.random_state)
        schemes = []

        if cfg.scheme_type == "quantile":
            for i in range(n_features):
                schemes.append(QuantileBinningScheme(feature_idx=i, n_bins=cfg.n_bins))
            return schemes

        if cfg.scheme_type == "kmeans":
            n_kmeans = max(1, n_features // 2)
            for _ in range(n_kmeans):
                n_select = max(2, rng.randint(2, n_features + 1)) if n_features >= 2 else 1
                n_select = min(n_select, n_features)
                feature_indices = rng.choice(n_features, size=n_select, replace=False)
                schemes.append(
                    KMeansScheme(
                        feature_indices=feature_indices,
                        n_clusters=min(cfg.n_kmeans_clusters, 50),
                        random_state=rng.randint(0, 10000),
                    )
                )
            return schemes

        if cfg.scheme_type == "tree":
            for i in range(cfg.n_tree_partitions):
                schemes.append(
                    DecisionTreeScheme(
                        max_depth=cfg.tree_max_depth,
                        random_state=cfg.random_state + i,
                    )
                )
            return schemes

        raise ValueError("scheme_type must be one of {'quantile', 'kmeans', 'tree'}")


def _build_supervised_dataset_from_splits(
    X_train_id,
    X_eval_id,
    X_train_ood,
    X_eval_ood,
    config: SingleSchemeConfig,
    balance_strategy="none",
):
    repr_model = SingleSchemePartitionOOD(config)
    repr_model.fit(X_train_id)

    R_train_id = repr_model.transform(X_train_id)
    R_test_id = repr_model.transform(X_eval_id)
    R_train_ood = repr_model.transform(X_train_ood)
    R_test_ood = repr_model.transform(X_eval_ood)

    if config.use_raw_features:
        raw_scaler = StandardScaler()
        raw_train_id = raw_scaler.fit_transform(X_train_id)
        raw_test_id = raw_scaler.transform(X_eval_id)
        raw_train_ood = raw_scaler.transform(X_train_ood)
        raw_test_ood = raw_scaler.transform(X_eval_ood)

        R_train_id = np.hstack([R_train_id, raw_train_id])
        R_test_id = np.hstack([R_test_id, raw_test_id])
        R_train_ood = np.hstack([R_train_ood, raw_train_ood])
        R_test_ood = np.hstack([R_test_ood, raw_test_ood])

    X_train = np.vstack([R_train_id, R_train_ood])
    y_train = np.concatenate([np.zeros(len(R_train_id)), np.ones(len(R_train_ood))])
    X_test = np.vstack([R_test_id, R_test_ood])
    y_test = np.concatenate([np.zeros(len(R_test_id)), np.ones(len(R_test_ood))])

    repr_scaler = StandardScaler()
    repr_scaler.fit(R_train_id)
    X_train = repr_scaler.transform(X_train)
    X_test = repr_scaler.transform(X_test)

    X_train, y_train = _balance_training_data(
        X_train, y_train, strategy=balance_strategy, random_state=config.random_state
    )

    return X_train, X_test, y_train, y_test


def _balance_training_data(X_train, y_train, strategy="none", random_state=42):
    if strategy == "none":
        return X_train, y_train

    y_train = np.asarray(y_train)
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        return X_train, y_train

    rng = np.random.RandomState(random_state)
    if strategy == "oversample":
        target = max(len(idx0), len(idx1))
        idx0_new = rng.choice(idx0, size=target, replace=True)
        idx1_new = rng.choice(idx1, size=target, replace=True)
    elif strategy == "undersample":
        target = min(len(idx0), len(idx1))
        idx0_new = rng.choice(idx0, size=target, replace=False)
        idx1_new = rng.choice(idx1, size=target, replace=False)
    else:
        raise ValueError("balance_strategy must be one of {'none', 'oversample', 'undersample'}")

    idx = np.concatenate([idx0_new, idx1_new])
    rng.shuffle(idx)
    return X_train[idx], y_train[idx]


def _build_supervised_dataset(
    ID_data,
    OOD_data,
    config: SingleSchemeConfig,
    test_size=0.3,
    balance_strategy="none",
):
    id_values = ID_data.values if isinstance(ID_data, pd.DataFrame) else ID_data
    ood_values = OOD_data.values if isinstance(OOD_data, pd.DataFrame) else OOD_data

    X_train_id, X_test_id = train_test_split(
        id_values,
        test_size=test_size,
        random_state=config.random_state,
    )
    X_train_ood, X_test_ood = train_test_split(
        ood_values,
        test_size=test_size,
        random_state=config.random_state,
    )

    return _build_supervised_dataset_from_splits(
        X_train_id=X_train_id,
        X_eval_id=X_test_id,
        X_train_ood=X_train_ood,
        X_eval_ood=X_test_ood,
        config=config,
        balance_strategy=balance_strategy,
    )


def _classification_metrics(y_test, y_pred, y_proba):
    return {
        "roc_auc": sk_metrics.roc_auc_score(y_test, y_proba),
        "pr_auc": sk_metrics.average_precision_score(y_test, y_proba),
        "accuracy": sk_metrics.accuracy_score(y_test, y_pred),
        "precision": sk_metrics.precision_score(y_test, y_pred),
        "recall": sk_metrics.recall_score(y_test, y_pred),
        "f1": sk_metrics.f1_score(y_test, y_pred),
    }


def _train_and_score(X_train, X_test, y_train, y_test, classifier_name="lr", random_state=42):
    if classifier_name == "lr":
        clf = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=random_state,
            class_weight="balanced",
        )
    elif classifier_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            criterion="log_loss",
            class_weight="balanced",
        )
    elif classifier_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=random_state,
        )
    else:
        raise ValueError("classifier_name must be one of {'lr', 'rf', 'mlp'}")

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    return _classification_metrics(y_test, y_pred, y_proba)


def run_single_parameter_sweep(
    ID_data,
    OOD_data,
    scheme_type,
    parameter_name,
    parameter_values,
    fixed_params=None,
    test_size=0.3,
    random_state=42,
    meta_features=None,
    use_raw_features=True,
    classifier_name="mlp",
    balance_strategy="oversample",
    cv_folds=5,
):
    """
    Evaluate quality dependence on one hyperparameter for a chosen scheme.
    """
    fixed_params = fixed_params or {}
    rows = []

    for value in parameter_values:
        cfg_kwargs = {
            "scheme_type": scheme_type,
            "random_state": random_state,
            "meta_features": meta_features,
            "use_raw_features": use_raw_features,
            **fixed_params,
        }
        cfg_kwargs[parameter_name] = value
        config = SingleSchemeConfig(**cfg_kwargs)

        if cv_folds and cv_folds > 1:
            fold_metrics, summary = cross_validate_single_scheme(
                ID_data=ID_data,
                OOD_data=OOD_data,
                config=config,
                classifier_name=classifier_name,
                n_splits=cv_folds,
                random_state=random_state,
                balance_strategy=balance_strategy,
            )
            row = {
                parameter_name: value,
                "roc_auc": float(summary["roc_auc_mean"]),
                "pr_auc": float(summary["pr_auc_mean"]),
                "accuracy": float(summary["accuracy_mean"]),
                "precision": float(summary["precision_mean"]),
                "recall": float(summary["recall_mean"]),
                "f1": float(summary["f1_mean"]),
                "roc_auc_std": float(summary["roc_auc_std"]),
                "pr_auc_std": float(summary["pr_auc_std"]),
                "f1_std": float(summary["f1_std"]),
            }
            rows.append(row)
        else:
            X_train, X_test, y_train, y_test = _build_supervised_dataset(
                ID_data=ID_data,
                OOD_data=OOD_data,
                config=config,
                test_size=test_size,
                balance_strategy=balance_strategy,
            )
            metrics = _train_and_score(
                X_train, X_test, y_train, y_test,
                classifier_name=classifier_name,
                random_state=random_state,
            )
            metrics[parameter_name] = value
            rows.append(metrics)

    return pd.DataFrame(rows).sort_values(by=parameter_name).reset_index(drop=True)


def cross_validate_single_scheme(
    ID_data,
    OOD_data,
    config: SingleSchemeConfig,
    classifier_name="mlp",
    n_splits=5,
    random_state=42,
    balance_strategy="oversample",
):
    '''
    K-fold evaluation for a single scheme configuration.
    '''
    id_values = ID_data.values if isinstance(ID_data, pd.DataFrame) else ID_data
    ood_values = OOD_data.values if isinstance(OOD_data, pd.DataFrame) else OOD_data

    kf_id = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf_ood = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, ((id_train_idx, id_test_idx), (ood_train_idx, ood_test_idx)) in enumerate(
        zip(kf_id.split(id_values), kf_ood.split(ood_values)),
        start=1
    ):
        X_train, X_test, y_train, y_test = _build_supervised_dataset_from_splits(
            X_train_id=id_values[id_train_idx],
            X_eval_id=id_values[id_test_idx],
            X_train_ood=ood_values[ood_train_idx],
            X_eval_ood=ood_values[ood_test_idx],
            config=config,
            balance_strategy=balance_strategy,
        )
        metrics = _train_and_score(
            X_train, X_test, y_train, y_test,
            classifier_name=classifier_name,
            random_state=random_state + fold,
        )
        metrics["fold"] = fold
        rows.append(metrics)

    fold_metrics = pd.DataFrame(rows)
    summary = {
        "roc_auc_mean": fold_metrics["roc_auc"].mean(),
        "roc_auc_std": fold_metrics["roc_auc"].std(ddof=0),
        "pr_auc_mean": fold_metrics["pr_auc"].mean(),
        "pr_auc_std": fold_metrics["pr_auc"].std(ddof=0),
        "accuracy_mean": fold_metrics["accuracy"].mean(),
        "accuracy_std": fold_metrics["accuracy"].std(ddof=0),
        "precision_mean": fold_metrics["precision"].mean(),
        "precision_std": fold_metrics["precision"].std(ddof=0),
        "recall_mean": fold_metrics["recall"].mean(),
        "recall_std": fold_metrics["recall"].std(ddof=0),
        "f1_mean": fold_metrics["f1"].mean(),
        "f1_std": fold_metrics["f1"].std(ddof=0),
    }
    return fold_metrics, summary


def plot_cv_sweep(df, parameter_name, metric="roc_auc", dataset_name=""):
    '''
    Visualize parameter sweep with CV mean/std.
    '''
    mean_col = metric
    std_col = f"{metric}_std"
    if mean_col not in df.columns:
        raise ValueError(f"'{mean_col}' not found in DataFrame")

    plt.figure(figsize=(8, 5))
    plt.plot(df[parameter_name], df[mean_col], marker="o", linewidth=2)
    if std_col in df.columns:
        low = df[mean_col] - df[std_col]
        high = df[mean_col] + df[std_col]
        plt.fill_between(df[parameter_name], low, high, alpha=0.2)
    plt.title(f"{metric.upper()} vs {parameter_name} | {dataset_name}")
    plt.xlabel(parameter_name)
    plt.ylabel(metric.upper())
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def learn_hyperparameters_with_validation(
    ID_data,
    OOD_data,
    scheme_type,
    search_grid,
    test_size=0.3,
    val_size=0.3,
    random_state=42,
    meta_features=None,
    use_raw_features=True,
    classifier_name="mlp",
    balance_strategy="oversample",
):
    """
    Learn hyperparameters with leakage-safe nested validation:
    1) Outer split: train/test
    2) Inner split on outer-train: train/val for hyperparameter search
    3) Refit with best params on outer-train, evaluate on outer-test
    """
    id_values = ID_data.values if isinstance(ID_data, pd.DataFrame) else ID_data
    ood_values = OOD_data.values if isinstance(OOD_data, pd.DataFrame) else OOD_data

    id_train_outer, id_test_outer = train_test_split(
        id_values, test_size=test_size, random_state=random_state
    )
    ood_train_outer, ood_test_outer = train_test_split(
        ood_values, test_size=test_size, random_state=random_state
    )

    id_train_inner, id_val_inner = train_test_split(
        id_train_outer, test_size=val_size, random_state=random_state
    )
    ood_train_inner, ood_val_inner = train_test_split(
        ood_train_outer, test_size=val_size, random_state=random_state
    )

    param_names = list(search_grid.keys())
    best_params = None
    best_val_auc = -np.inf
    search_rows = []

    for values in product(*(search_grid[k] for k in param_names)):
        params = dict(zip(param_names, values))

        config = SingleSchemeConfig(
            scheme_type=scheme_type,
            random_state=random_state,
            meta_features=meta_features,
            use_raw_features=use_raw_features,
            **params,
        )

        X_train, X_val, y_train, y_val = _build_supervised_dataset_from_splits(
            X_train_id=id_train_inner,
            X_eval_id=id_val_inner,
            X_train_ood=ood_train_inner,
            X_eval_ood=ood_val_inner,
            config=config,
            balance_strategy=balance_strategy,
        )
        val_metrics = _train_and_score(
            X_train, X_val, y_train, y_val,
            classifier_name=classifier_name,
            random_state=random_state,
        )

        row = dict(params)
        row.update(val_metrics)
        search_rows.append(row)

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_params = params

    best_config = SingleSchemeConfig(
        scheme_type=scheme_type,
        random_state=random_state,
        meta_features=meta_features,
        use_raw_features=use_raw_features,
        **best_params,
    )

    X_train_final, X_test_final, y_train_final, y_test_final = _build_supervised_dataset_from_splits(
        X_train_id=id_train_outer,
        X_eval_id=id_test_outer,
        X_train_ood=ood_train_outer,
        X_eval_ood=ood_test_outer,
        config=best_config,
        balance_strategy=balance_strategy,
    )
    test_metrics = _train_and_score(
        X_train_final, X_test_final, y_train_final, y_test_final,
        classifier_name=classifier_name,
        random_state=random_state,
    )

    return {
        "best_params": best_params,
        "best_validation_roc_auc": best_val_auc,
        "test_metrics": test_metrics,
        "search_table": pd.DataFrame(search_rows).sort_values(
            by=["roc_auc"], ascending=False
        ),
    }


def _build_full_ensemble_dataset_from_splits(
    X_train_id,
    X_eval_id,
    X_train_ood,
    X_eval_ood,
    n_bins=10,
    n_kmeans_clusters=10,
    n_tree_partitions=5,
    tree_max_depth=3,
    random_state=42,
    meta_features=None,
    use_raw_features=True,
    balance_strategy="none",
):
    model = EnsemblePartitionOOD(
        n_bins=n_bins,
        n_kmeans_clusters=n_kmeans_clusters,
        n_tree_partitions=n_tree_partitions,
        tree_max_depth=tree_max_depth,
        random_state=random_state,
        meta_features=meta_features,
    )
    model.fit(X_train_id)

    R_train_id = model.transform(X_train_id)
    R_eval_id = model.transform(X_eval_id)
    R_train_ood = model.transform(X_train_ood)
    R_eval_ood = model.transform(X_eval_ood)

    if use_raw_features:
        raw_scaler = StandardScaler()
        raw_train_id = raw_scaler.fit_transform(X_train_id)
        raw_eval_id = raw_scaler.transform(X_eval_id)
        raw_train_ood = raw_scaler.transform(X_train_ood)
        raw_eval_ood = raw_scaler.transform(X_eval_ood)

        R_train_id = np.hstack([R_train_id, raw_train_id])
        R_eval_id = np.hstack([R_eval_id, raw_eval_id])
        R_train_ood = np.hstack([R_train_ood, raw_train_ood])
        R_eval_ood = np.hstack([R_eval_ood, raw_eval_ood])

    X_train = np.vstack([R_train_id, R_train_ood])
    y_train = np.concatenate([np.zeros(len(R_train_id)), np.ones(len(R_train_ood))])
    X_eval = np.vstack([R_eval_id, R_eval_ood])
    y_eval = np.concatenate([np.zeros(len(R_eval_id)), np.ones(len(R_eval_ood))])

    repr_scaler = StandardScaler()
    repr_scaler.fit(R_train_id)
    X_train = repr_scaler.transform(X_train)
    X_eval = repr_scaler.transform(X_eval)

    X_train, y_train = _balance_training_data(
        X_train, y_train, strategy=balance_strategy, random_state=random_state
    )

    return X_train, X_eval, y_train, y_eval


def learn_all_schemes_hyperparameters_with_validation(
    ID_data,
    OOD_data,
    search_grid,
    test_size=0.3,
    val_size=0.3,
    random_state=42,
    meta_features=None,
    use_raw_features=True,
    classifier_name="mlp",
    balance_strategy="oversample",
):
    """
    Jointly learn all partition hyperparameters (quantile + kmeans + tree)
    using a leakage-safe nested validation protocol.
    """
    id_values = ID_data.values if isinstance(ID_data, pd.DataFrame) else ID_data
    ood_values = OOD_data.values if isinstance(OOD_data, pd.DataFrame) else OOD_data

    id_train_outer, id_test_outer = train_test_split(
        id_values, test_size=test_size, random_state=random_state
    )
    ood_train_outer, ood_test_outer = train_test_split(
        ood_values, test_size=test_size, random_state=random_state
    )

    id_train_inner, id_val_inner = train_test_split(
        id_train_outer, test_size=val_size, random_state=random_state
    )
    ood_train_inner, ood_val_inner = train_test_split(
        ood_train_outer, test_size=val_size, random_state=random_state
    )

    param_names = list(search_grid.keys())
    best_params = None
    best_val_auc = -np.inf
    search_rows = []

    for values in product(*(search_grid[k] for k in param_names)):
        params = dict(zip(param_names, values))
        X_train, X_val, y_train, y_val = _build_full_ensemble_dataset_from_splits(
            X_train_id=id_train_inner,
            X_eval_id=id_val_inner,
            X_train_ood=ood_train_inner,
            X_eval_ood=ood_val_inner,
            random_state=random_state,
            meta_features=meta_features,
            use_raw_features=use_raw_features,
            balance_strategy=balance_strategy,
            **params,
        )
        val_metrics = _train_and_score(
            X_train, X_val, y_train, y_val,
            classifier_name=classifier_name,
            random_state=random_state,
        )

        row = dict(params)
        row.update(val_metrics)
        search_rows.append(row)

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_params = params

    X_train_final, X_test_final, y_train_final, y_test_final = _build_full_ensemble_dataset_from_splits(
        X_train_id=id_train_outer,
        X_eval_id=id_test_outer,
        X_train_ood=ood_train_outer,
        X_eval_ood=ood_test_outer,
        random_state=random_state,
        meta_features=meta_features,
        use_raw_features=use_raw_features,
        balance_strategy=balance_strategy,
        **best_params,
    )

    lr_test_metrics = _train_and_score(
        X_train_final,
        X_test_final,
        y_train_final,
        y_test_final,
        classifier_name="lr",
        random_state=random_state,
    )
    rf_test_metrics = _train_and_score(
        X_train_final,
        X_test_final,
        y_train_final,
        y_test_final,
        classifier_name="rf",
        random_state=random_state,
    )
    mlp_test_metrics = _train_and_score(
        X_train_final,
        X_test_final,
        y_train_final,
        y_test_final,
        classifier_name="mlp",
        random_state=random_state,
    )

    learned_rows = [
        {
            "classifier": "Logistic Regression",
            **best_params,
            **lr_test_metrics,
        },
        {
            "classifier": "Tree (Random Forest)",
            **best_params,
            **rf_test_metrics,
        },
        {
            "classifier": "MLP",
            **best_params,
            **mlp_test_metrics,
        },
    ]

    return {
        "best_params": best_params,
        "best_validation_roc_auc": best_val_auc,
        "learned_metrics_df": pd.DataFrame(learned_rows),
        "search_table": pd.DataFrame(search_rows).sort_values(
            by=["roc_auc"], ascending=False
        ),
    }
