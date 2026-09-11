# Meta-features (36)

The representation uses 36 descriptors, grouped below.  `pairwise_pearson_corr_mean_abs` is an alias of `mean_abs_correlation` and is dropped.  `mean_radius` and `max_radius` are legacy descriptors and are not part of the taxonomy.

`local_outlier_factor_score` is the ratio of the query kNN distance to the mean kNN distance in the cell.  `mahalanobis` uses the diagonal of the cell covariance.

## 1) Point-based (8)

| Name | Description |
|---|---|
| `dist_to_mean` | Distance of the point to the cell mean |
| `normalized_dist` | Distance to the mean, scaled by the cell radius |
| `cosine_dist` | Cosine distance to the cell mean |
| `dist_to_median` | Distance of the point to the cell median |
| `log_likelihood` | Diagonal-Gaussian log-likelihood of the point in the cell |
| `local_outlier_factor_score` | Query kNN distance divided by the cell mean kNN distance |
| `knn_distance_k` | Distance to the k-th neighbour in the cell |
| `out_of_range_count` | Number of coordinates outside the cell min–max range |

## 2) Marginal statistical (7)

| Name | Description |
|---|---|
| `mean_norm` | Norm of the cell mean vector |
| `std_norm` | Norm of the cell standard-deviation vector |
| `skew_norm` | Norm of per-feature skewness |
| `kurtosis_norm` | Norm of per-feature kurtosis |
| `median_abs_deviation_norm` | Norm of the MAD vector |
| `iqr_norm` | Norm of the interquartile-range vector |
| `trimmed_mean_norm` | Norm of the 10%-trimmed mean |

## 3) Marginal informational (5)

| Name | Description |
|---|---|
| `cell_entropy` | Entropy of feature histograms inside the cell |
| `marginal_entropy_mean` | Mean marginal entropy |
| `marginal_entropy_std` | Standard deviation of marginal entropies |
| `marginal_kl_to_reference_mean` | Mean KL divergence to the ID reference |
| `quantile_surprisal_mean` | Mean quantile surprisal |

## 4) Interaction statistical (6)

| Name | Description |
|---|---|
| `mahalanobis` | Diagonal Mahalanobis distance to the cell mean |
| `covariance_trace` | Trace of the cell covariance |
| `covariance_logdet` | Log-determinant of the cell covariance |
| `mean_abs_correlation` | Mean absolute Pearson correlation |
| `pairwise_spearman_corr_mean_abs` | Mean absolute Spearman correlation |
| `covariance_condition_number` | Condition number of the cell covariance |

## 5) Interaction informational (5)

| Name | Description |
|---|---|
| `pairwise_mutual_info_mean` | Mean pairwise mutual information |
| `pairwise_mutual_info_max` | Maximum pairwise mutual information |
| `total_correlation` | Gaussian total correlation |
| `joint_entropy_pairwise_mean` | Mean pairwise joint entropy |
| `pairwise_js_divergence_mean` | Mean pairwise Jensen–Shannon divergence |

## 6) Base (5)

| Name | Description |
|---|---|
| `log_count` | Log of the number of ID training points in the cell |
| `density` | Cell size as a fraction of the ID training sample |
| `n_beyond_2std` | Number of coordinates beyond two standard deviations |
| `partition_agreement_count` | Number of schemes that place the point in a dense cell |
| `leaf_depth` | Tree-leaf depth; zero for non-tree schemes |
