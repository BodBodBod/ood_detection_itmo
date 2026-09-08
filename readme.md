# OOD Detection in Tabular Data

This repository collects three OOD detection methods on tabular data. All of them use the same seven benchmark datasets in `data/`, but they answer different questions at different levels.

| Folder | Question | Method |
| --- | --- | --- |
| `sample_wise_ae_ks/` | Is this **batch** OOD? | Autoencoder + KS-test + Monte Carlo |
| `point_wise_ae_distances/` | Is this **row** OOD? | AE reconstruction error metrics + classifier |
| `point_wise_partition_meta/` | Is this **row** OOD? | Partition ensemble + meta-features |

```text
OOD_detection/
├── data/
├── sample_wise_ae_ks/
├── point_wise_ae_distances/
├── point_wise_partition_meta/
└── readme.md
```

---

## [DRAFT] point_wise_partition_meta

This part detects OOD at the level of **individual rows**. The idea is to split the feature space into many cells with several partition schemes, describe each point by statistics of its cell (meta-features), and train a classifier on top.

### Layout

```text
point_wise_partition_meta/
├── partition_ood.py
├── partitioning_scheme_learning.py
├── metacharacteristics_taxonomy_light.md
├── partition_ood_experiments.ipynb
├── partition_ood_id_ood_experiments.ipynb
├── partitioning_scheme_experiment.ipynb
├── metacharacteristics_exclude_include_experiment.ipynb
└── metacharacteristics_significance_experiment.ipynb
```

Shared data sits one level up:

```text
data/
├── taxi_source.csv / taxi_target.csv
├── electricity_source.csv / electricity_target.csv
├── income_source.csv / income_target.csv
├── mv_x6_source.csv / mv_x6_target.csv
├── diabites_source.csv / diabites_target.csv
├── california_source.csv / california_target.csv
└── acs_accidents_source.csv / acs_accidents_target.csv
```

`*_source` files are in-distribution (ID), `*_target` files are out-of-distribution (OOD).  
`partition_ood.load_dataset()` resolves paths from the repo root, so `data/` must stay next to the method folders.

### Core module: `partition_ood.py`

**Partition schemes**

- `QuantileBinningScheme` - quantile bins along one feature.
- `KMeansScheme` - KMeans clusters on a random feature subset.
- `DecisionTreeScheme` - a tree fit on synthetic labels; each leaf is a cell.

`EnsemblePartitionOOD` combines all three families: quantile schemes for every feature, several k-means schemes, and several tree schemes.

**Meta-features**

The registry is `ALL_META_FEATURES`. Experiments use a list close to the light taxonomy: point-based distances and outlier counts, marginal and feature-interaction statistics, and basic cell support measures (`log_count`, `density`, and related fields).

**Training and evaluation**

- `build_ood_dataset` - train/test split, fit partitions, build the feature matrix.
- `partition_fit_data='id'` or `'id+ood'` - whether partition schemes are fit on ID only or on ID plus OOD (test rows never enter the fit step).
- `cross_validate_ood_classifiers` / `full_pipeline` - cross-validation, class balancing, metrics (ROC-AUC, PR-AUC, F1, and others), plots.

### Scheme hyperparameters: `partitioning_scheme_learning.py`

A thin layer on top of `partition_ood.py` for experiments that vary **one scheme family** instead of the full ensemble.

- `SingleSchemePartitionOOD` - quantile-only, k-means-only, or tree-only mode.
- Sweeps over `n_bins`, `n_kmeans_clusters`, `tree_max_depth`, `n_tree_partitions`.
- An optional validation split for hyperparameter search.

Used by `partitioning_scheme_experiment.ipynb`.

### Taxonomy: `metacharacteristics_taxonomy_light.md`

Groups meta-features for the significance experiment:

1. **Point-based** - where the point sits inside its cell.
2. **Sample-based / Marginal-char** - per-feature stats and entropies inside the cell.
3. **Sample-based / Feature-interaction** - covariances, correlations, mutual information.
4. **Base** - cell size, density, tree depth, agreement across schemes.

Only the light taxonomy file is included here; notebooks use it as the reference list of feature groups.

### Notebooks

**`partition_ood_experiments.ipynb`**  
Baseline run of the full ensemble on all datasets. Starting point for other comparisons.

**`partition_ood_id_ood_experiments.ipynb`**  
Compares fitting partitions on ID only vs ID + OOD (train splits only). Ran on all seven datasets.

**`partitioning_scheme_experiment.ipynb`**  
Measures single scheme families and their hyperparameters (quantile `n_bins`, k-means cluster count, tree depth and number of trees, plus a learned-hyperparameter variant). The run stopped early on Taxi, so results are incomplete.

**`metacharacteristics_exclude_include_experiment.ipynb`**  
Three input setups: meta-features only, meta-features plus raw features, raw features only. Partial runs on Taxi, Electricity, and Income.

**`metacharacteristics_significance_experiment.ipynb`**  
Compares **groups** of meta-features from the light taxonomy (not one feature at a time). Each group gets a CV score; output is a table and a ROC-AUC bar chart.

### Quick start

Run notebooks from the repo root or from `point_wise_partition_meta/` if `sys.path` includes the parent directory.

```python
from point_wise_partition_meta.partition_ood import load_dataset, full_pipeline

ID, OOD = load_dataset('Taxi')
results = full_pipeline(ID, OOD, n_splits=5, use_cv=True)
```

Hyperparameter grids and meta-feature lists are defined inside the notebooks.

---

## sample_wise_ae_ks

This part answers: **is an entire sample (a batch of rows) OOD?**  
It trains an autoencoder on ID data, compares reconstruction error distributions with a KS-test, and validates the method with Monte Carlo simulations.

### Layout

```text
sample_wise_ae_ks/
├── model.py
├── pipeline.py
├── autoencode_real_data.ipynb
└── ood_bucket_experiments.ipynb
```

### Pipeline

- **Autoencoder (AE):** trained to reconstruct ID data.
- **Reconstruction error:** scalar RMSE per row, aggregated into a sample-level error distribution.
- **KS-test:** compares the error distribution of a test sample to a held-out ID control sample. A low p-value suggests an OOD sample.
- **Monte Carlo:** checks false positive rate control and sensitivity in `pipeline.py`.

`autoencode_real_data.ipynb` also runs the AE point-level experiments from `point_wise_ae_distances/` (same trained model, different downstream step).

Datasets load from `../data/`.

---

## point_wise_ae_distances

This part answers: **is a single row OOD?** using autoencoder reconstruction error.

It shares the AE core (`model.py`, `pipeline.py`) with `sample_wise_ae_ks/`, but builds a separate feature vector from the full error vector $(x - \hat{x})$ and trains a classifier on top.

### Layout

```text
point_wise_ae_distances/
├── point_wise_classification.py
├── AE_ROC_modeling.py
└── images/decision_tree/
```

### Pipeline

- **Error vector:** full reconstruction residual per point.
- **Distance metrics:** L1, L2, L-infinity, RMSE, Mahalanobis distance, counts above 1/2/3 standard deviations, and related scores.
- **Classifier:** decision tree, random forest, or logistic regression on these metrics.
- **Synthetic OOD:** optional training mode with Gaussian noise added to ID error features.

`AE_ROC_modeling.py` links sample-level thresholds to point-level detection via ROC-based calibration.

Main experiments live in `../sample_wise_ae_ks/autoencode_real_data.ipynb`.

### Datasets

Seven tabular benchmarks with fixed ID/OOD splits from [ITMO NSS LAB](https://github.com/ITMO-NSS-team/OOD_Tab_Evaluation/tree/main).

| Dataset | ID samples | OOD samples | Features | Numerical / categorical |
| :-- | :--: | :--: | :--: | :-- |
| **Taxi** | 10,000 | 10,000 | 7 | N: 7 |
| **Electricity** | 9,986 | 10,014 | 6 | N: 6 |
| **Income** | 20,380 | 9,782 | 12 | N: 4, C: 8 |
| **MV X6** | 20,384 | 20,384 | 9 | N: 6, C: 3 |
| **Diabites** | 34,288 | 1,500 | 183 | N: 10, C: 173 |
| **California** | 10,315 | 10,319 | 7 | N: 6, C: 1 |
| **ACS Accidents** | 22,653 | 3,955 | 45 | N: 45 |

### How to run

Open `sample_wise_ae_ks/autoencode_real_data.ipynb` or `sample_wise_ae_ks/ood_bucket_experiments.ipynb` with working directory set to `sample_wise_ae_ks/`.  
Point-level modules in `point_wise_ae_distances/` import the AE pipeline from the sibling folder automatically.
