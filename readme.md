# OOD Detection in Tabular Data

This repository collects three OOD detection methods on tabular data. They share the benchmark tables in `data/` and answer different questions at different resolutions.

| Folder | Question | Method |
| --- | --- | --- |
| `sample_wise_ae_ks/` | Is this **batch** OOD? | Autoencoder + KS-test + Monte Carlo |
| `point_wise_ae_distances/` | Is this **row** OOD? | AE reconstruction error metrics + classifier |
| `point_wise_partition_meta/` | Is this **row** OOD? | Partition ensemble + 36 meta-features |

```text
OOD_detection/
├── data/
├── sample_wise_ae_ks/
├── point_wise_ae_distances/
├── point_wise_partition_meta/
├── results/
├── requirements.txt
└── readme.md
```

Install with `pip install -r requirements.txt`. Reported five-fold CSVs for the partition method sit in `results/`. Reproduce them from the repository root:

```bash
python -m point_wise_partition_meta.run_experiments --task representation
python -m point_wise_partition_meta.run_experiments --task id_ood
python -m point_wise_partition_meta.run_experiments --task importance
python -m point_wise_partition_meta.run_experiments --task groups
```

Default datasets are Taxi, Electricity, Income, MVx6, California, and ACS Accidents. Taxi is loaded with **6** numeric features.

---

## point_wise_partition_meta

This part detects OOD at the level of **individual rows**. The feature space is split into cells with several partition schemes. Each point is described by 36 statistics of its ID reference cell (meta-features). A classifier is trained on the concatenated descriptor vector.

### Layout

```text
point_wise_partition_meta/
├── partition_ood.py
├── partitioning_scheme_learning.py
├── run_experiments.py
├── metacharacteristics_taxonomy_light.md
└── (legacy notebooks)
```

Shared data sits one level up. `*_source` files are in-distribution (ID), `*_target` files are out-of-distribution (OOD). `partition_ood.load_dataset()` resolves paths from the repo root.

The canonical taxonomy is `TAXONOMY_META_FEATURES` in `partition_ood.py` (36 descriptors in six groups). See `metacharacteristics_taxonomy_light.md`.

### Quick start

```python
from point_wise_partition_meta.partition_ood import load_dataset, full_pipeline

ID, OOD = load_dataset('Taxi')
results = full_pipeline(ID, OOD, cv_folds=5)
```

### Scheme families: `partitioning_scheme_learning.py`

`SingleSchemePartitionOOD` keeps only quantile, k-means, or tree partitions and uses the same 36-feature extractor as the full ensemble.

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

Six tabular benchmarks with fixed ID/OOD splits from [ITMO NSS LAB](https://github.com/ITMO-NSS-team/OOD_Tab_Evaluation/tree/main). These are the pairs used in the reported experiments.

| Dataset | ID samples | OOD samples | Features | Numerical / categorical |
| :-- | :--: | :--: | :--: | :-- |
| **Taxi** | 10,000 | 10,000 | 6 | N: 6 |
| **Electricity** | 9,986 | 10,014 | 6 | N: 6 |
| **Income** | 20,380 | 9,782 | 12 | N: 4, C: 8 |
| **MV X6** | 20,384 | 20,384 | 9 | N: 6, C: 3 |
| **California** | 10,315 | 10,319 | 7 | N: 6, C: 1 |
| **ACS Accidents** | 22,653 | 3,955 | 45 | N: 45 |

### How to run

Open `sample_wise_ae_ks/autoencode_real_data.ipynb` or `sample_wise_ae_ks/ood_bucket_experiments.ipynb` with working directory set to `sample_wise_ae_ks/`.  
Point-level modules in `point_wise_ae_distances/` import the AE pipeline from the sibling folder automatically.
