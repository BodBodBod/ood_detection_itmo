'''
Reproducible experiment runner for the partition-meta OOD detector.

Run from the repository root, for example:

    python -m point_wise_partition_meta.run_experiments --task representation
    python -m point_wise_partition_meta.run_experiments --task id_ood
    python -m point_wise_partition_meta.run_experiments --task scheme_families
    python -m point_wise_partition_meta.run_experiments --task groups
    python -m point_wise_partition_meta.run_experiments --task importance
    python -m point_wise_partition_meta.run_experiments --task seeds
    python -m point_wise_partition_meta.run_experiments --task all

CSV files are written to ``results/``. The default dataset list matches the
reported experiments (Taxi, Electricity, Income, MVx6, California, ACS Accidents).
'''

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from point_wise_partition_meta.partition_ood import (
    DATASETS,
    META_FEATURE_GROUPS,
    META_FEATURE_GROUP_LABELS,
    TAXONOMY_META_FEATURES,
    _build_ood_dataset_from_splits,
    aggregate_importance,
    cross_validate_ood_classifiers,
    load_dataset,
    train_ood_classifiers,
    wilcoxon_paired,
)
from point_wise_partition_meta.partitioning_scheme_learning import (
    SingleSchemeConfig,
    _build_supervised_dataset_from_splits,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'results'

REPRESENTATIONS = {
    'raw': {'include_meta': False, 'use_raw_features': True},
    'meta': {'include_meta': True, 'use_raw_features': False},
    'meta_raw': {'include_meta': True, 'use_raw_features': True},
}


def _to_numpy(frame):
    return frame.values if hasattr(frame, 'values') else np.asarray(frame)


def _save(frame, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    frame.to_csv(path, index=False)
    print(f'wrote {path}')
    return path


def _summarize(fold_metrics, keys):
    grouped = fold_metrics.groupby(keys, as_index=False).agg(
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        pr_auc_mean=('pr_auc', 'mean'),
        pr_auc_std=('pr_auc', 'std'),
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        classifier_spread=('roc_auc', lambda s: float(s.max() - s.min()) if len(s) else np.nan),
    )
    return grouped


def _cv_one_config(
    ID_data,
    OOD_data,
    *,
    n_splits,
    random_state,
    include_mlp,
    include_meta,
    use_raw_features,
    partition_fit_data='id',
    meta_features=None,
):
    fold_metrics, summary, _ = cross_validate_ood_classifiers(
        ID_data=ID_data,
        OOD_data=OOD_data,
        n_splits=n_splits,
        random_state=random_state,
        use_raw_features=use_raw_features,
        include_meta=include_meta,
        meta_features=meta_features,
        partition_fit_data=partition_fit_data,
        include_mlp=include_mlp,
    )
    return fold_metrics, summary


def run_representation(datasets, n_splits, random_state, include_mlp):
    fold_rows = []
    for dataset in datasets:
        print(f'\n=== representation | {dataset} ===')
        ID_data, OOD_data = load_dataset(dataset)
        for name, cfg in REPRESENTATIONS.items():
            fold_metrics, _ = _cv_one_config(
                ID_data,
                OOD_data,
                n_splits=n_splits,
                random_state=random_state,
                include_mlp=include_mlp,
                meta_features=TAXONOMY_META_FEATURES,
                **cfg,
            )
            fold_metrics = fold_metrics.copy()
            fold_metrics['dataset'] = dataset
            fold_metrics['representation'] = name
            fold_metrics['seed'] = random_state
            fold_rows.append(fold_metrics)
            print(fold_metrics.groupby('model')[['roc_auc', 'pr_auc', 'f1']].mean())

    folds = pd.concat(fold_rows, ignore_index=True)
    _save(folds, 'representation_folds.csv')
    summary = _summarize(folds, ['dataset', 'representation', 'model'])
    _save(summary, 'representation_summary.csv')
    _save(_significance_table(folds), 'representation_significance.csv')
    _save(_linearization_table(summary), 'linearization_summary.csv')
    return folds


def _significance_table(folds, metric='roc_auc'):
    rows = []
    pairs = [('meta', 'raw'), ('meta_raw', 'raw'), ('meta_raw', 'meta')]
    for dataset in folds['dataset'].unique():
        for model in folds['model'].unique():
            for left_name, right_name in pairs:
                left = folds[
                    (folds['dataset'] == dataset)
                    & (folds['model'] == model)
                    & (folds['representation'] == left_name)
                ].sort_values('fold')[metric]
                right = folds[
                    (folds['dataset'] == dataset)
                    & (folds['model'] == model)
                    & (folds['representation'] == right_name)
                ].sort_values('fold')[metric]
                if len(left) == 0 or len(right) == 0 or len(left) != len(right):
                    continue
                test = wilcoxon_paired(left, right)
                rows.append({
                    'dataset': dataset,
                    'model': model,
                    'metric': metric,
                    'left': left_name,
                    'right': right_name,
                    **test,
                })
    return pd.DataFrame(rows)


def _linearization_table(summary):
    rows = []
    for dataset in summary['dataset'].unique():
        for representation in summary['representation'].unique():
            chunk = summary[
                (summary['dataset'] == dataset)
                & (summary['representation'] == representation)
            ]
            if chunk.empty:
                continue
            lr = chunk.loc[chunk['model'] == 'Logistic Regression', 'roc_auc_mean']
            spread = float(
                chunk['roc_auc_mean'].max() - chunk['roc_auc_mean'].min()
            )
            rows.append({
                'dataset': dataset,
                'representation': representation,
                'lr_roc_auc': float(lr.iloc[0]) if len(lr) else np.nan,
                'best_roc_auc': float(chunk['roc_auc_mean'].max()),
                'classifier_spread': spread,
            })
    return pd.DataFrame(rows)


def run_id_ood(datasets, n_splits, random_state, include_mlp):
    fold_rows = []
    for dataset in datasets:
        print(f'\n=== id vs id+ood | {dataset} ===')
        ID_data, OOD_data = load_dataset(dataset)
        for fit_mode in ('id', 'id+ood'):
            fold_metrics, _ = _cv_one_config(
                ID_data,
                OOD_data,
                n_splits=n_splits,
                random_state=random_state,
                include_mlp=include_mlp,
                include_meta=True,
                use_raw_features=False,
                partition_fit_data=fit_mode,
                meta_features=TAXONOMY_META_FEATURES,
            )
            fold_metrics = fold_metrics.copy()
            fold_metrics['dataset'] = dataset
            fold_metrics['partition_fit_data'] = fit_mode
            fold_metrics['seed'] = random_state
            fold_rows.append(fold_metrics)
            print(fit_mode)
            print(fold_metrics.groupby('model')[['roc_auc', 'pr_auc', 'f1']].mean())

    folds = pd.concat(fold_rows, ignore_index=True)
    _save(folds, 'id_ood_folds.csv')
    _save(_summarize(folds, ['dataset', 'partition_fit_data', 'model']), 'id_ood_summary.csv')

    sig_rows = []
    for dataset in folds['dataset'].unique():
        for model in folds['model'].unique():
            left = folds[
                (folds['dataset'] == dataset)
                & (folds['model'] == model)
                & (folds['partition_fit_data'] == 'id+ood')
            ].sort_values('fold')['roc_auc']
            right = folds[
                (folds['dataset'] == dataset)
                & (folds['model'] == model)
                & (folds['partition_fit_data'] == 'id')
            ].sort_values('fold')['roc_auc']
            if len(left) != len(right) or len(left) == 0:
                continue
            test = wilcoxon_paired(left, right)
            sig_rows.append({
                'dataset': dataset,
                'model': model,
                'left': 'id+ood',
                'right': 'id',
                **test,
            })
    _save(pd.DataFrame(sig_rows), 'id_ood_significance.csv')
    return folds


def run_scheme_families(datasets, n_splits, random_state, include_mlp):
    fold_rows = []
    for dataset in datasets:
        print(f'\n=== scheme families | {dataset} ===')
        ID_data, OOD_data = load_dataset(dataset)
        id_values = _to_numpy(ID_data)
        ood_values = _to_numpy(OOD_data)
        kf_id = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        kf_ood = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold_idx, ((id_tr, id_te), (ood_tr, ood_te)) in enumerate(
            zip(kf_id.split(id_values), kf_ood.split(ood_values)),
            start=1,
        ):
            for scheme_type in ('quantile', 'kmeans', 'tree'):
                config = SingleSchemeConfig(
                    scheme_type=scheme_type,
                    random_state=random_state + fold_idx,
                    meta_features=TAXONOMY_META_FEATURES,
                    use_raw_features=False,
                )
                X_train, X_test, y_train, y_test = _build_supervised_dataset_from_splits(
                    X_train_id=id_values[id_tr],
                    X_eval_id=id_values[id_te],
                    X_train_ood=ood_values[ood_tr],
                    X_eval_ood=ood_values[ood_te],
                    config=config,
                    balance_strategy='oversample',
                )
                metrics_df, _ = train_ood_classifiers(
                    X_train, X_test, y_train, y_test,
                    include_mlp=include_mlp,
                    random_state=random_state + fold_idx,
                )
                for model_name, row in metrics_df.iterrows():
                    fold_rows.append({
                        'dataset': dataset,
                        'scheme_type': scheme_type,
                        'fold': fold_idx,
                        'model': model_name,
                        'roc_auc': float(row['roc-auc']),
                        'pr_auc': float(row['pr-auc']),
                        'f1': float(row['f1-score']),
                        'seed': random_state,
                    })
        print(pd.DataFrame(fold_rows).query('dataset == @dataset').groupby(
            ['scheme_type', 'model']
        )['roc_auc'].mean())

    folds = pd.DataFrame(fold_rows)
    _save(folds, 'scheme_family_folds.csv')
    _save(_summarize(folds, ['dataset', 'scheme_type', 'model']), 'scheme_family_summary.csv')
    return folds


def run_groups(datasets, n_splits, random_state, include_mlp):
    fold_rows = []
    for dataset in datasets:
        print(f'\n=== meta-feature groups | {dataset} ===')
        ID_data, OOD_data = load_dataset(dataset)
        configs = {'all': TAXONOMY_META_FEATURES, **META_FEATURE_GROUPS}
        for group_name, features in configs.items():
            fold_metrics, _ = _cv_one_config(
                ID_data,
                OOD_data,
                n_splits=n_splits,
                random_state=random_state,
                include_mlp=include_mlp,
                include_meta=True,
                use_raw_features=False,
                meta_features=features,
            )
            fold_metrics = fold_metrics.copy()
            fold_metrics['dataset'] = dataset
            fold_metrics['group'] = group_name
            fold_metrics['group_label'] = META_FEATURE_GROUP_LABELS.get(group_name, group_name)
            fold_metrics['n_meta_features'] = len(features)
            fold_metrics['seed'] = random_state
            fold_rows.append(fold_metrics)
            print(group_name, fold_metrics.groupby('model')['roc_auc'].mean().to_dict())

    folds = pd.concat(fold_rows, ignore_index=True)
    _save(folds, 'group_ablation_folds.csv')
    _save(_summarize(folds, ['dataset', 'group', 'model']), 'group_ablation_summary.csv')
    return folds


def run_importance(datasets, random_state, include_mlp):
    rows = []
    descriptor_rows = []
    group_rows = []
    for dataset in datasets:
        print(f'\n=== importance | {dataset} ===')
        ID_data, OOD_data = load_dataset(dataset)
        id_values = _to_numpy(ID_data)
        ood_values = _to_numpy(OOD_data)
        kf_id = KFold(n_splits=5, shuffle=True, random_state=random_state)
        kf_ood = KFold(n_splits=5, shuffle=True, random_state=random_state)
        (id_tr, id_te), (ood_tr, ood_te) = next(zip(
            kf_id.split(id_values), kf_ood.split(ood_values)
        ))
        X_train, X_test, y_train, y_test, model, _ = _build_ood_dataset_from_splits(
            X_train_id=id_values[id_tr],
            X_test_id=id_values[id_te],
            X_train_ood=ood_values[ood_tr],
            X_test_ood=ood_values[ood_te],
            random_state=random_state,
            use_raw_features=False,
            include_meta=True,
            meta_features=TAXONOMY_META_FEATURES,
            partition_fit_data='id',
        )
        _, classifiers = train_ood_classifiers(
            X_train, X_test, y_train, y_test,
            include_mlp=False,
            random_state=random_state,
        )
        names = model.feature_names()
        sources = {
            'Logistic Regression': np.abs(classifiers['Logistic Regression'].coef_[0]),
            'Random Forest': classifiers['Random Forest'].feature_importances_,
        }
        for model_name, values in sources.items():
            per_feature, by_descriptor, by_group = aggregate_importance(names, values)
            per_feature['dataset'] = dataset
            per_feature['model'] = model_name
            by_descriptor['dataset'] = dataset
            by_descriptor['model'] = model_name
            by_group['dataset'] = dataset
            by_group['model'] = model_name
            rows.append(per_feature)
            descriptor_rows.append(by_descriptor)
            group_rows.append(by_group)
            print(model_name)
            print(by_group.to_string(index=False))

    _save(pd.concat(rows, ignore_index=True), 'importance_features.csv')
    _save(pd.concat(descriptor_rows, ignore_index=True), 'importance_descriptors.csv')
    _save(pd.concat(group_rows, ignore_index=True), 'importance_groups.csv')


def run_seeds(datasets, n_splits, seeds, include_mlp):
    fold_rows = []
    for seed in seeds:
        print(f'\n=== seed {seed} ===')
        for dataset in datasets:
            ID_data, OOD_data = load_dataset(dataset)
            fold_metrics, _ = _cv_one_config(
                ID_data,
                OOD_data,
                n_splits=n_splits,
                random_state=seed,
                include_mlp=include_mlp,
                include_meta=True,
                use_raw_features=False,
                meta_features=TAXONOMY_META_FEATURES,
            )
            fold_metrics = fold_metrics.copy()
            fold_metrics['dataset'] = dataset
            fold_metrics['seed'] = seed
            fold_rows.append(fold_metrics)

    folds = pd.concat(fold_rows, ignore_index=True)
    _save(folds, 'seed_folds.csv')
    _save(_summarize(folds, ['dataset', 'model', 'seed']), 'seed_summary.csv')
    across = (
        folds.groupby(['dataset', 'model'], as_index=False)
        .agg(
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            n=('roc_auc', 'size'),
        )
    )
    _save(across, 'seed_across_summary.csv')
    return folds


def parse_args():
    parser = argparse.ArgumentParser(description='Partition-meta OOD experiments')
    parser.add_argument(
        '--task',
        required=True,
        choices=[
            'representation',
            'id_ood',
            'scheme_families',
            'groups',
            'importance',
            'seeds',
            'all',
        ],
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['Taxi', 'Electricity', 'Income', 'MVx6', 'California', 'ACS Accidents'],
        choices=list(DATASETS),
    )
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', default='42,123,7')
    parser.add_argument('--no-mlp', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    include_mlp = not args.no_mlp
    datasets = args.datasets
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in {'representation', 'all'}:
        run_representation(datasets, args.n_splits, args.seed, include_mlp)
    if args.task in {'id_ood', 'all'}:
        run_id_ood(datasets, args.n_splits, args.seed, include_mlp)
    if args.task in {'scheme_families', 'all'}:
        run_scheme_families(datasets, args.n_splits, args.seed, include_mlp)
    if args.task in {'groups', 'all'}:
        run_groups(datasets, args.n_splits, args.seed, include_mlp)
    if args.task in {'importance', 'all'}:
        run_importance(datasets, args.seed, include_mlp)
    if args.task in {'seeds', 'all'}:
        run_seeds(datasets, args.n_splits, seeds, include_mlp)


if __name__ == '__main__':
    main()
