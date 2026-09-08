from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial import cKDTree

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics as sk_metrics

from matplotlib import pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')


# =====================================================================
# Meta-feature registry
# =====================================================================

ALL_META_FEATURES = [
    # --- point-based ---
    'dist_to_mean',
    'normalized_dist',
    'cosine_dist',
    'dist_to_median',
    'log_likelihood',
    'local_outlier_factor_score',
    'knn_distance_k',

    # --- sample-based / marginal-char / statistical ---
    'mean_norm',
    'std_norm',
    'skew_norm',
    'kurtosis_norm',
    'median_abs_deviation_norm',
    'iqr_norm',
    'trimmed_mean_norm',
    # Backward-compatible features already used in experiments
    'mean_radius',
    'max_radius',

    # --- sample-based / marginal-char / informational ---
    'cell_entropy',
    'marginal_entropy_mean',
    'marginal_entropy_std',
    'marginal_kl_to_reference_mean',
    'quantile_surprisal_mean',

    # --- sample-based / feature-interaction / statistical ---
    'mahalanobis',
    'covariance_trace',
    'covariance_logdet',
    'mean_abs_correlation',
    'pairwise_pearson_corr_mean_abs',
    'pairwise_spearman_corr_mean_abs',
    'covariance_condition_number',

    # --- sample-based / feature-interaction / informational ---
    'pairwise_mutual_info_mean',
    'pairwise_mutual_info_max',
    'total_correlation',
    'joint_entropy_pairwise_mean',
    'pairwise_js_divergence_mean',

    # --- base ---
    'log_count',
    'density',
    'out_of_range_count',
    'n_beyond_2std',
    'partition_agreement_count',
    'leaf_depth',
]

DEFAULT_META_FEATURES = [
    'log_count',
    'density',
    'mean_norm',
    'std_norm',
    'dist_to_mean',
]


# =====================================================================
# Partitioning Schemes
# =====================================================================

class PartitionScheme:
    '''Base class for a partitioning scheme.'''

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        '''Returns group labels for each sample.'''
        raise NotImplementedError


class QuantileBinningScheme(PartitionScheme):
    '''Partition by quantile binning of a single feature.'''

    def __init__(self, feature_idx, n_bins=10):
        self.feature_idx = feature_idx
        self.n_bins = n_bins
        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode='ordinal', strategy='quantile',
            subsample=None
        )

    def fit(self, X):
        self.discretizer.fit(X[:, self.feature_idx:self.feature_idx + 1])
        return self

    def predict(self, X):
        return self.discretizer.transform(
            X[:, self.feature_idx:self.feature_idx + 1]
        ).ravel().astype(int)


class KMeansScheme(PartitionScheme):
    '''Partition using KMeans on selected features.'''

    def __init__(self, feature_indices, n_clusters=10, random_state=42):
        self.feature_indices = feature_indices
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )

    def fit(self, X):
        self.kmeans.fit(X[:, self.feature_indices])
        return self

    def predict(self, X):
        return self.kmeans.predict(X[:, self.feature_indices])


class DecisionTreeScheme(PartitionScheme):
    '''Partition using decision tree leaf assignments on synthetic task.'''

    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = None

    def fit(self, X):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        y_synthetic = rng.randint(0, 2, size=n)
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.tree.fit(X, y_synthetic)
        return self

    def predict(self, X):
        return self.tree.apply(X)


# =====================================================================
# Ensemble-of-Partitions OOD Detector
# =====================================================================

class EnsemblePartitionOOD:
    '''
    Ensemble-of-Partitions OOD detector.

    Constructs M partitioning schemes, computes per-cell statistics on
    fit data, and builds a derived representation for each sample based on
    its partition assignments and cell statistics.
    '''

    def __init__(
        self,
        n_bins=10,
        n_kmeans_clusters=10,
        n_tree_partitions=5,
        tree_max_depth=3,
        random_state=42,
        meta_features=None,
    ):
        self.n_bins = n_bins
        self.n_kmeans_clusters = n_kmeans_clusters
        self.n_tree_partitions = n_tree_partitions
        self.tree_max_depth = tree_max_depth
        self.random_state = random_state
        self.meta_features = meta_features if meta_features is not None else DEFAULT_META_FEATURES

        # Validate meta feature names
        unknown = set(self.meta_features) - set(ALL_META_FEATURES)
        if unknown:
            raise ValueError(
                f"Unknown meta features: {unknown}. "
                f"Available: {ALL_META_FEATURES}"
            )

        self.scaler = StandardScaler()
        self.schemes = []
        self.cell_stats = []
        self.scheme_density_thresholds = []
        self.n_features_ = None

    def _build_schemes(self, n_features):
        '''Create the partitioning schemes.'''
        schemes = []
        rng = np.random.RandomState(self.random_state)

        # 1. Quantile binning on each feature
        for i in range(n_features):
            schemes.append(
                QuantileBinningScheme(feature_idx=i, n_bins=self.n_bins)
            )

        # 2. KMeans on random subsets of features
        n_kmeans = max(1, n_features // 2)
        for _ in range(n_kmeans):
            n_select = max(2, rng.randint(2, n_features + 1))
            feature_indices = rng.choice(n_features, size=n_select, replace=False)
            schemes.append(KMeansScheme(
                feature_indices=feature_indices,
                n_clusters=min(self.n_kmeans_clusters, 50),
                random_state=rng.randint(0, 10000)
            ))

        # 3. Decision tree partitions
        for i in range(self.n_tree_partitions):
            schemes.append(DecisionTreeScheme(
                max_depth=self.tree_max_depth,
                random_state=self.random_state + i
            ))

        return schemes

    def _compute_cell_stats(self, X, labels):
        '''
        Compute statistics for each cell in a partition.

        Only the statistics required by ``self.meta_features`` are computed
        (along with their dependencies such as ``mean``, ``std``, etc.).
        '''
        mf = set(self.meta_features)
        stats = {}
        unique_labels = np.unique(labels)
        total = X.shape[0]
        n_feat = X.shape[1]

        # Shared constants
        eps = 1e-12

        # Pre-compute which intermediate stats are needed
        need_mean = bool(mf & {
            'mean_norm', 'mean_radius', 'max_radius', 'dist_to_mean',
            'normalized_dist', 'mahalanobis', 'cosine_dist',
            'log_likelihood', 'n_beyond_2std', 'local_outlier_factor_score',
        })
        need_std = bool(mf & {
            'std_norm', 'mahalanobis', 'log_likelihood', 'n_beyond_2std',
            'local_outlier_factor_score',
        })
        need_median = 'dist_to_median' in mf
        need_minmax = 'out_of_range_count' in mf
        need_skew = 'skew_norm' in mf
        need_kurtosis = 'kurtosis_norm' in mf
        need_radius = bool(mf & {'mean_radius', 'max_radius', 'normalized_dist'})
        need_mad = 'median_abs_deviation_norm' in mf
        need_iqr = 'iqr_norm' in mf
        need_trimmed_mean = 'trimmed_mean_norm' in mf
        need_entropy_stats = bool(mf & {
            'cell_entropy', 'marginal_entropy_mean', 'marginal_entropy_std',
            'marginal_kl_to_reference_mean', 'quantile_surprisal_mean',
        })
        need_cov = bool(mf & {
            'covariance_trace', 'covariance_logdet', 'mean_abs_correlation',
            'pairwise_pearson_corr_mean_abs', 'pairwise_spearman_corr_mean_abs',
            'covariance_condition_number', 'total_correlation',
        })
        need_pairwise_info = bool(mf & {
            'pairwise_mutual_info_mean', 'pairwise_mutual_info_max',
            'joint_entropy_pairwise_mean', 'pairwise_js_divergence_mean',
        })
        need_knn = bool(mf & {'knn_distance_k', 'local_outlier_factor_score'})

        # Reference histograms and quantile bins used by informational features
        global_hist = None
        global_quantile_edges = None
        if need_entropy_stats:
            n_bins_hist = max(5, min(20, int(np.sqrt(max(2, total)))))
            global_hist = []
            global_quantile_edges = []
            for j in range(n_feat):
                col = X[:, j]
                cmin, cmax = float(col.min()), float(col.max())
                if np.isclose(cmin, cmax):
                    edges = np.array([cmin - 1.0, cmax + 1.0], dtype=np.float64)
                else:
                    edges = np.linspace(cmin, cmax, n_bins_hist + 1)

                h, _ = np.histogram(col, bins=edges)
                p = h.astype(np.float64)
                p = p / max(p.sum(), 1.0)
                global_hist.append((edges, p))

                q_edges = np.quantile(col, np.linspace(0.0, 1.0, 11))
                q_edges = np.unique(q_edges)
                if q_edges.shape[0] < 2:
                    q_edges = np.array([cmin - 1.0, cmax + 1.0], dtype=np.float64)
                global_quantile_edges.append(q_edges)

        # Quantized matrix for pairwise-information features
        X_quant = None
        if need_pairwise_info:
            X_quant = np.zeros_like(X, dtype=np.int32)
            for j in range(n_feat):
                col = X[:, j]
                edges = np.quantile(col, np.linspace(0.0, 1.0, 11))
                edges = np.unique(edges)
                if edges.shape[0] < 2:
                    X_quant[:, j] = 0
                else:
                    X_quant[:, j] = np.clip(np.digitize(col, edges[1:-1]), 0, 9)

        for g in unique_labels:
            mask = labels == g
            X_g = X[mask]
            count = X_g.shape[0]

            s = {'count': count}

            # --- Basic moments ---
            if need_mean:
                s['mean'] = X_g.mean(axis=0) if count > 0 else np.zeros(n_feat)
            if need_std:
                s['std'] = X_g.std(axis=0) if count > 1 else np.zeros(n_feat)
            if need_median:
                s['median'] = np.median(X_g, axis=0) if count > 0 else np.zeros(n_feat)
            if need_minmax and count > 0:
                s['min'] = X_g.min(axis=0)
                s['max'] = X_g.max(axis=0)
            elif need_minmax:
                s['min'] = np.zeros(n_feat)
                s['max'] = np.zeros(n_feat)

            # --- Derived cell-level scalars ---
            if 'log_count' in mf:
                s['log_count'] = np.log1p(count)
            if 'density' in mf:
                s['density'] = count / total
            if 'mean_norm' in mf:
                s['mean_norm'] = np.linalg.norm(s['mean'])
            if 'std_norm' in mf:
                s['std_norm'] = np.linalg.norm(s['std'])

            # Distances inside cell (for mean_radius / max_radius)
            if need_radius and count > 0:
                dists = np.linalg.norm(X_g - s['mean'], axis=1)
                s['mean_radius'] = float(dists.mean()) if count > 0 else 0.0
                s['max_radius'] = float(dists.max()) if count > 0 else 0.0
            elif need_radius:
                s['mean_radius'] = 0.0
                s['max_radius'] = 0.0

            if need_mad:
                if count > 0:
                    med = np.median(X_g, axis=0)
                    mad = np.median(np.abs(X_g - med), axis=0)
                    s['median_abs_deviation_norm'] = float(np.linalg.norm(mad))
                else:
                    s['median_abs_deviation_norm'] = 0.0

            if need_iqr:
                if count > 0:
                    q75 = np.percentile(X_g, 75, axis=0)
                    q25 = np.percentile(X_g, 25, axis=0)
                    s['iqr_norm'] = float(np.linalg.norm(q75 - q25))
                else:
                    s['iqr_norm'] = 0.0

            if need_trimmed_mean:
                if count > 0:
                    tmean = sp_stats.trim_mean(X_g, proportiontocut=0.1, axis=0)
                    s['trimmed_mean_norm'] = float(np.linalg.norm(tmean))
                else:
                    s['trimmed_mean_norm'] = 0.0

            if need_entropy_stats:
                entropies = []
                kl_vals = []
                surprisal_vals = []

                for j in range(n_feat):
                    vals = X_g[:, j]
                    edges, p_ref = global_hist[j]
                    hist, _ = np.histogram(vals, bins=edges)
                    p_cell = hist.astype(np.float64)
                    p_cell = p_cell / max(p_cell.sum(), 1.0)

                    ent = -np.sum(p_cell[p_cell > 0] * np.log(p_cell[p_cell > 0] + eps))
                    entropies.append(float(ent))

                    if 'marginal_kl_to_reference_mean' in mf:
                        ratio = (p_cell + eps) / (p_ref + eps)
                        kl_vals.append(float(np.sum(p_cell * np.log(ratio))))

                    if 'quantile_surprisal_mean' in mf:
                        q_edges = global_quantile_edges[j]
                        q_bins = np.clip(np.digitize(vals, q_edges[1:-1]), 0, len(q_edges) - 2)
                        q_counts = np.bincount(q_bins, minlength=len(q_edges) - 1).astype(np.float64)
                        q_probs = q_counts / max(q_counts.sum(), 1.0)
                        if len(vals) > 0:
                            surprisal_vals.append(float(np.mean(-np.log(q_probs[q_bins] + eps))))

                if 'cell_entropy' in mf:
                    s['cell_entropy'] = float(np.mean(entropies)) if entropies else 0.0
                if 'marginal_entropy_mean' in mf:
                    s['marginal_entropy_mean'] = float(np.mean(entropies)) if entropies else 0.0
                if 'marginal_entropy_std' in mf:
                    s['marginal_entropy_std'] = float(np.std(entropies)) if entropies else 0.0
                if 'marginal_kl_to_reference_mean' in mf:
                    s['marginal_kl_to_reference_mean'] = float(np.mean(kl_vals)) if kl_vals else 0.0
                if 'quantile_surprisal_mean' in mf:
                    s['quantile_surprisal_mean'] = float(np.mean(surprisal_vals)) if surprisal_vals else 0.0

            if need_skew:
                if count > 2:
                    skew = sp_stats.skew(X_g, axis=0, nan_policy='omit')
                    skew = np.nan_to_num(skew, nan=0.0)
                    s['skew_norm'] = float(np.linalg.norm(skew))
                else:
                    s['skew_norm'] = 0.0

            if need_kurtosis:
                if count > 3:
                    kurt = sp_stats.kurtosis(X_g, axis=0, nan_policy='omit')
                    kurt = np.nan_to_num(kurt, nan=0.0)
                    s['kurtosis_norm'] = float(np.linalg.norm(kurt))
                else:
                    s['kurtosis_norm'] = 0.0

            # Diagonal inverse covariance (for Mahalanobis)
            if 'mahalanobis' in mf:
                std_safe = np.where(s['std'] > 1e-12, s['std'], 1.0)
                s['diag_inv_cov'] = 1.0 / (std_safe ** 2)

            if need_cov:
                if count > 1:
                    cov = np.cov(X_g, rowvar=False)
                    if cov.ndim == 0:
                        cov = np.array([[float(cov)]], dtype=np.float64)
                else:
                    cov = np.eye(n_feat, dtype=np.float64) * eps

                cov_safe = cov + np.eye(cov.shape[0], dtype=np.float64) * eps

                if 'covariance_trace' in mf:
                    s['covariance_trace'] = float(np.trace(cov_safe))
                if 'covariance_logdet' in mf:
                    sign, ldet = np.linalg.slogdet(cov_safe)
                    s['covariance_logdet'] = float(ldet if sign > 0 else np.log(eps))
                if 'covariance_condition_number' in mf:
                    s['covariance_condition_number'] = float(np.linalg.cond(cov_safe))
                if 'total_correlation' in mf:
                    diag_cov = np.diag(np.diag(cov_safe))
                    sign_d, ldet_d = np.linalg.slogdet(diag_cov)
                    sign_c, ldet_c = np.linalg.slogdet(cov_safe)
                    tc = 0.5 * ((ldet_d if sign_d > 0 else np.log(eps)) - (ldet_c if sign_c > 0 else np.log(eps)))
                    s['total_correlation'] = float(max(0.0, tc))

                if 'mean_abs_correlation' in mf or 'pairwise_pearson_corr_mean_abs' in mf:
                    if count > 1:
                        corr = np.corrcoef(X_g, rowvar=False)
                        if corr.ndim == 0:
                            corr = np.array([[1.0]], dtype=np.float64)
                        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                        iu = np.triu_indices(corr.shape[0], k=1)
                        vals = np.abs(corr[iu])
                        mean_abs_corr = float(vals.mean()) if vals.size > 0 else 0.0
                    else:
                        mean_abs_corr = 0.0

                    if 'mean_abs_correlation' in mf:
                        s['mean_abs_correlation'] = mean_abs_corr
                    if 'pairwise_pearson_corr_mean_abs' in mf:
                        s['pairwise_pearson_corr_mean_abs'] = mean_abs_corr

                if 'pairwise_spearman_corr_mean_abs' in mf:
                    if count > 1:
                        spearman = sp_stats.spearmanr(X_g, axis=0).correlation
                        if spearman is None or np.isscalar(spearman):
                            s['pairwise_spearman_corr_mean_abs'] = 0.0
                        else:
                            spearman = np.nan_to_num(spearman, nan=0.0, posinf=0.0, neginf=0.0)
                            iu = np.triu_indices(spearman.shape[0], k=1)
                            vals = np.abs(spearman[iu])
                            s['pairwise_spearman_corr_mean_abs'] = float(vals.mean()) if vals.size > 0 else 0.0
                    else:
                        s['pairwise_spearman_corr_mean_abs'] = 0.0

            if need_pairwise_info:
                idx = np.where(mask)[0]
                Xg_q = X_quant[idx] if idx.size > 0 else np.empty((0, n_feat), dtype=np.int32)
                mi_vals = []
                joint_entropy_vals = []
                js_vals = []
                for a in range(n_feat):
                    for b in range(a + 1, n_feat):
                        qa = Xg_q[:, a]
                        qb = Xg_q[:, b]
                        if qa.size == 0:
                            continue

                        if 'pairwise_mutual_info_mean' in mf or 'pairwise_mutual_info_max' in mf:
                            mi_vals.append(float(sk_metrics.mutual_info_score(qa, qb)))

                        if 'joint_entropy_pairwise_mean' in mf or 'pairwise_js_divergence_mean' in mf:
                            n_bin = max(int(max(np.max(qa), np.max(qb)) + 1), 2)
                            joint = np.zeros((n_bin, n_bin), dtype=np.float64)
                            for ia, ib in zip(qa, qb):
                                joint[ia, ib] += 1.0
                            p_joint = joint / max(joint.sum(), 1.0)
                            p_joint_nz = p_joint[p_joint > 0]
                            joint_entropy_vals.append(float(-np.sum(p_joint_nz * np.log(p_joint_nz + eps))))

                            if 'pairwise_js_divergence_mean' in mf:
                                pa = p_joint.sum(axis=1, keepdims=True)
                                pb = p_joint.sum(axis=0, keepdims=True)
                                p_prod = pa @ pb
                                m_mix = 0.5 * (p_joint + p_prod)
                                js = 0.5 * np.sum(p_joint * np.log((p_joint + eps) / (m_mix + eps)))
                                js += 0.5 * np.sum(p_prod * np.log((p_prod + eps) / (m_mix + eps)))
                                js_vals.append(float(js))

                if 'pairwise_mutual_info_mean' in mf:
                    s['pairwise_mutual_info_mean'] = float(np.mean(mi_vals)) if mi_vals else 0.0
                if 'pairwise_mutual_info_max' in mf:
                    s['pairwise_mutual_info_max'] = float(np.max(mi_vals)) if mi_vals else 0.0
                if 'joint_entropy_pairwise_mean' in mf:
                    s['joint_entropy_pairwise_mean'] = float(np.mean(joint_entropy_vals)) if joint_entropy_vals else 0.0
                if 'pairwise_js_divergence_mean' in mf:
                    s['pairwise_js_divergence_mean'] = float(np.mean(js_vals)) if js_vals else 0.0

            if need_knn:
                if count > 0:
                    s['kdtree'] = cKDTree(X_g)
                    k_ref = min(6, count)
                    d_ref, _ = s['kdtree'].query(X_g, k=k_ref)
                    if np.ndim(d_ref) == 1:
                        d_ref = d_ref[:, None]
                    # Exclude self-distance in the first column when available
                    neigh = d_ref[:, 1:] if d_ref.shape[1] > 1 else d_ref
                    s['mean_knn_dist_ref'] = float(np.mean(neigh)) if neigh.size > 0 else 0.0
                else:
                    s['kdtree'] = None
                    s['mean_knn_dist_ref'] = 0.0

            stats[g] = s

        return stats

    def _extract_features(self, X, scheme_idx, labels, agreement_counts=None, leaf_depths=None):
        '''
        Extract per-sample features based on cell assignment.

        The returned feature vector for each sample has length
        ``len(self.meta_features)`` and respects the ordering of
        ``self.meta_features``.
        '''
        stats = self.cell_stats[scheme_idx]
        mf_list = self.meta_features
        n_mf = len(mf_list)
        n_samples = X.shape[0]

        result = np.zeros((n_samples, n_mf), dtype=np.float64)

        for i, label in enumerate(labels):
            if label not in stats:
                continue  # leave zeros

            s = stats[label]
            x = X[i]

            # Lazily computed point-level values (only when needed)
            _dist_to_mean = None
            _diff = None

            for j, name in enumerate(mf_list):
                # --- cell-level scalars ---
                if name == 'log_count':
                    result[i, j] = s['log_count']
                elif name == 'density':
                    result[i, j] = s['density']
                elif name == 'mean_norm':
                    result[i, j] = s['mean_norm']
                elif name == 'std_norm':
                    result[i, j] = s['std_norm']
                elif name == 'mean_radius':
                    result[i, j] = s['mean_radius']
                elif name == 'max_radius':
                    result[i, j] = s['max_radius']
                elif name == 'cell_entropy':
                    result[i, j] = s['cell_entropy']
                elif name == 'skew_norm':
                    result[i, j] = s['skew_norm']
                elif name == 'kurtosis_norm':
                    result[i, j] = s['kurtosis_norm']
                elif name == 'median_abs_deviation_norm':
                    result[i, j] = s.get('median_abs_deviation_norm', 0.0)
                elif name == 'iqr_norm':
                    result[i, j] = s.get('iqr_norm', 0.0)
                elif name == 'trimmed_mean_norm':
                    result[i, j] = s.get('trimmed_mean_norm', 0.0)
                elif name == 'marginal_entropy_mean':
                    result[i, j] = s.get('marginal_entropy_mean', 0.0)
                elif name == 'marginal_entropy_std':
                    result[i, j] = s.get('marginal_entropy_std', 0.0)
                elif name == 'marginal_kl_to_reference_mean':
                    result[i, j] = s.get('marginal_kl_to_reference_mean', 0.0)
                elif name == 'quantile_surprisal_mean':
                    result[i, j] = s.get('quantile_surprisal_mean', 0.0)
                elif name == 'covariance_trace':
                    result[i, j] = s.get('covariance_trace', 0.0)
                elif name == 'covariance_logdet':
                    result[i, j] = s.get('covariance_logdet', 0.0)
                elif name == 'mean_abs_correlation':
                    result[i, j] = s.get('mean_abs_correlation', 0.0)
                elif name == 'pairwise_pearson_corr_mean_abs':
                    result[i, j] = s.get('pairwise_pearson_corr_mean_abs', 0.0)
                elif name == 'pairwise_spearman_corr_mean_abs':
                    result[i, j] = s.get('pairwise_spearman_corr_mean_abs', 0.0)
                elif name == 'covariance_condition_number':
                    result[i, j] = s.get('covariance_condition_number', 0.0)
                elif name == 'pairwise_mutual_info_mean':
                    result[i, j] = s.get('pairwise_mutual_info_mean', 0.0)
                elif name == 'pairwise_mutual_info_max':
                    result[i, j] = s.get('pairwise_mutual_info_max', 0.0)
                elif name == 'total_correlation':
                    result[i, j] = s.get('total_correlation', 0.0)
                elif name == 'joint_entropy_pairwise_mean':
                    result[i, j] = s.get('joint_entropy_pairwise_mean', 0.0)
                elif name == 'pairwise_js_divergence_mean':
                    result[i, j] = s.get('pairwise_js_divergence_mean', 0.0)
                elif name == 'partition_agreement_count':
                    result[i, j] = float(agreement_counts[i]) if agreement_counts is not None else 0.0
                elif name == 'leaf_depth':
                    result[i, j] = float(leaf_depths[i]) if leaf_depths is not None else 0.0

                # --- point-level scalars ---
                elif name == 'dist_to_mean':
                    if _diff is None:
                        _diff = x - s['mean']
                    if _dist_to_mean is None:
                        _dist_to_mean = np.linalg.norm(_diff)
                    result[i, j] = _dist_to_mean

                elif name == 'normalized_dist':
                    if _diff is None:
                        _diff = x - s['mean']
                    if _dist_to_mean is None:
                        _dist_to_mean = np.linalg.norm(_diff)
                    mr = s['mean_radius']
                    result[i, j] = _dist_to_mean / mr if mr > 1e-12 else 0.0

                elif name == 'mahalanobis':
                    if _diff is None:
                        _diff = x - s['mean']
                    result[i, j] = float(np.sqrt(
                        np.sum(_diff ** 2 * s['diag_inv_cov'])
                    ))

                elif name == 'cosine_dist':
                    mn = s['mean']
                    denom = np.linalg.norm(x) * np.linalg.norm(mn)
                    if denom > 1e-12:
                        result[i, j] = 1.0 - np.dot(x, mn) / denom
                    else:
                        result[i, j] = 1.0

                elif name == 'dist_to_median':
                    result[i, j] = np.linalg.norm(x - s['median'])

                elif name == 'log_likelihood':
                    std_safe = np.where(s['std'] > 1e-12, s['std'], 1.0)
                    # Diagonal Gaussian log-likelihood (summed over features)
                    result[i, j] = float(np.sum(
                        -0.5 * ((x - s['mean']) / std_safe) ** 2
                        - np.log(std_safe + 1e-12)
                    ))

                elif name == 'out_of_range_count':
                    result[i, j] = float(np.sum(
                        (x < s['min']) | (x > s['max'])
                    ))

                elif name == 'n_beyond_2std':
                    std_safe = np.where(s['std'] > 1e-12, s['std'], 1.0)
                    result[i, j] = float(np.sum(
                        np.abs(x - s['mean']) > 2.0 * std_safe
                    ))
                elif name == 'knn_distance_k':
                    if s.get('kdtree') is None:
                        result[i, j] = 0.0
                    else:
                        k = min(5, s['count'])
                        d, _ = s['kdtree'].query(x, k=k)
                        if np.ndim(d) == 0:
                            result[i, j] = float(d)
                        elif len(d) == 0:
                            result[i, j] = 0.0
                        else:
                            result[i, j] = float(d[-1])
                elif name == 'local_outlier_factor_score':
                    if s.get('kdtree') is None:
                        result[i, j] = 0.0
                    else:
                        k = min(5, s['count'])
                        d, _ = s['kdtree'].query(x, k=k)
                        if np.ndim(d) == 0:
                            d_mean = float(d)
                        elif len(d) == 0:
                            d_mean = 0.0
                        else:
                            d_arr = np.asarray(d)
                            d_mean = float(np.mean(d_arr[1:])) if len(d_arr) > 1 else float(d_arr[0])
                        denom = s.get('mean_knn_dist_ref', 0.0)
                        result[i, j] = d_mean / denom if denom > 1e-12 else d_mean

        return result

    def fit(self, X_train):
        '''
        Fit the ensemble of partitions on training (ID) data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        '''
        X_scaled = self.scaler.fit_transform(X_train)
        self.n_features_ = X_scaled.shape[1]

        self.schemes = self._build_schemes(self.n_features_)

        self.cell_stats = []
        self.scheme_density_thresholds = []
        for scheme in self.schemes:
            scheme.fit(X_scaled)
            labels = scheme.predict(X_scaled)
            stats = self._compute_cell_stats(X_scaled, labels)
            self.cell_stats.append(stats)
            densities = [v.get('density', 0.0) for v in stats.values()]
            if len(densities) > 0:
                self.scheme_density_thresholds.append(float(np.median(densities)))
            else:
                self.scheme_density_thresholds.append(0.0)

        return self

    def transform(self, X):
        '''
        Transform data into the partition-based representation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        representation : np.ndarray
        '''
        X_scaled = self.scaler.transform(X)

        all_labels = [scheme.predict(X_scaled) for scheme in self.schemes]

        # Count how many schemes place a sample into relatively dense cells
        agreement_counts = None
        if 'partition_agreement_count' in self.meta_features:
            agreement_counts = np.zeros(X_scaled.shape[0], dtype=np.float64)
            for idx, labels in enumerate(all_labels):
                threshold = self.scheme_density_thresholds[idx]
                s_stats = self.cell_stats[idx]
                for i, lb in enumerate(labels):
                    density = s_stats.get(lb, {}).get('density', 0.0)
                    if density >= threshold:
                        agreement_counts[i] += 1.0

        all_features = []
        for idx, (scheme, labels) in enumerate(zip(self.schemes, all_labels)):
            leaf_depths = None
            if 'leaf_depth' in self.meta_features and isinstance(scheme, DecisionTreeScheme):
                tree = scheme.tree.tree_
                node_depth = np.zeros(shape=tree.node_count, dtype=np.int32)
                stack = [(0, 0)]
                while stack:
                    node_id, depth = stack.pop()
                    node_depth[node_id] = depth
                    left = tree.children_left[node_id]
                    right = tree.children_right[node_id]
                    if left != right:
                        stack.append((left, depth + 1))
                        stack.append((right, depth + 1))
                leaf_depths = node_depth[labels]

            features = self._extract_features(
                X_scaled, idx, labels,
                agreement_counts=agreement_counts,
                leaf_depths=leaf_depths
            )
            all_features.append(features)

        return np.hstack(all_features)

    def fit_transform(self, X_train):
        '''Fit and transform in one step.'''
        self.fit(X_train)
        return self.transform(X_train)


# =====================================================================
# Data Loading Helpers
# =====================================================================

DATASETS = {
    'Taxi': {
        'source': 'data/taxi_source.csv',
        'target': 'data/taxi_target.csv',
        'loader': lambda path: pd.read_csv(
            path, index_col=0, usecols=[f'{i}' for i in range(7)]
        ),
    },
    'Electricity': {
        'source': 'data/electricity_source.csv',
        'target': 'data/electricity_target.csv',
        'loader': lambda path: pd.read_csv(
            path,
            usecols=['period', 'nswprice', 'nswdemand',
                     'vicprice', 'vicdemand', 'transfer']
        ),
    },
    'Income': {
        'source': 'data/income_source.csv',
        'target': 'data/income_target.csv',
        'loader': lambda path: (
            lambda df: df[df.columns[2:-1]]
        )(pd.read_csv(path).reset_index()),
    },
    'MVx6': {
        'source': 'data/mv_x6_source.csv',
        'target': 'data/mv_x6_target.csv',
        'loader': lambda path: pd.read_csv(
            path,
            usecols=['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x10']
        ),
    },
    'Diabetes': {
        'source': 'data/diabites_source.csv',
        'target': 'data/diabites_target.csv',
        'loader': lambda path: (
            lambda df: df[df.columns[:-1]]
        )(pd.read_csv(path)),
        'ood_limit': 1500,
    },
    'California': {
        'source': 'data/california_source.csv',
        'target': 'data/california_target.csv',
        'loader': lambda path: (
            lambda df: df[df.columns[1:-1]]
        )(pd.read_csv(path)),
    },
    'ACS Accidents': {
        'source': 'data/acs_accidents_source.csv',
        'target': 'data/acs_accidents_target.csv',
        'loader': lambda path: (
            lambda df: df[df.columns[1:-1]]
        )(pd.read_csv(path)),
    },
}


def load_dataset(name, data_dir='data'):
    '''
    Load ID and OOD data for a given dataset name.

    Returns
    -------
    ID_data, OOD_data : pd.DataFrame
    '''
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")

    # Resolve data paths relative to the project root (parent of this file's directory)
    _project_root = Path(__file__).resolve().parent.parent

    cfg = DATASETS[name]
    ID_data = cfg['loader'](str(_project_root / cfg['source']))
    OOD_data = cfg['loader'](str(_project_root / cfg['target']))

    ood_limit = cfg.get('ood_limit', None)
    if ood_limit is not None:
        OOD_data = OOD_data.iloc[:ood_limit]

    return ID_data, OOD_data


def describe_dataset(ID_data, OOD_data):
    '''Print basic info about ID and OOD data.'''
    print(f'ID data shape:      {ID_data.shape}')
    print(f'OOD data shape:     {OOD_data.shape}')
    print(f'Number of features: {len(ID_data.columns)}')


def _balance_training_data(X_train, y_train, strategy='none', random_state=42):
    '''
    Balance classes in the training set.

    Parameters
    ----------
    strategy : {'none', 'oversample', 'undersample'}
    '''
    if strategy == 'none':
        return X_train, y_train

    y_train = np.asarray(y_train)
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        return X_train, y_train

    rng = np.random.RandomState(random_state)

    if strategy == 'oversample':
        target = max(len(idx0), len(idx1))
        idx0_new = rng.choice(idx0, size=target, replace=True)
        idx1_new = rng.choice(idx1, size=target, replace=True)
    elif strategy == 'undersample':
        target = min(len(idx0), len(idx1))
        idx0_new = rng.choice(idx0, size=target, replace=False)
        idx1_new = rng.choice(idx1, size=target, replace=False)
    else:
        raise ValueError("balance_strategy must be one of {'none', 'oversample', 'undersample'}")

    new_idx = np.concatenate([idx0_new, idx1_new])
    rng.shuffle(new_idx)
    return X_train[new_idx], y_train[new_idx]


def _build_ood_dataset_from_splits(
    X_train_id,
    X_test_id,
    X_train_ood,
    X_test_ood,
    random_state=42,
    n_bins=10,
    n_kmeans_clusters=10,
    n_tree_partitions=5,
    tree_max_depth=3,
    use_raw_features=False,
    meta_features=None,
    partition_fit_data='id',
    balance_strategy='oversample',
):
    '''
    Build train/test representation from predefined ID/OOD splits.
    '''
    if partition_fit_data not in {'id', 'id+ood'}:
        raise ValueError("partition_fit_data must be one of {'id', 'id+ood'}")

    model = EnsemblePartitionOOD(
        n_bins=n_bins,
        n_kmeans_clusters=n_kmeans_clusters,
        n_tree_partitions=n_tree_partitions,
        tree_max_depth=tree_max_depth,
        random_state=random_state,
        meta_features=meta_features,
    )

    X_partition_fit = X_train_id if partition_fit_data == 'id' else np.vstack([X_train_id, X_train_ood])

    model.fit(X_partition_fit)
    R_train_id = model.transform(X_train_id)
    R_test_id = model.transform(X_test_id)
    R_train_ood = model.transform(X_train_ood)
    R_test_ood = model.transform(X_test_ood)

    if use_raw_features:
        raw_scaler = StandardScaler()
        raw_train_id = raw_scaler.fit_transform(X_train_id)
        raw_test_id = raw_scaler.transform(X_test_id)
        raw_train_ood = raw_scaler.transform(X_train_ood)
        raw_test_ood = raw_scaler.transform(X_test_ood)

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
        X_train, y_train, strategy=balance_strategy, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, model, repr_scaler


# =====================================================================
# OOD Dataset Construction
# =====================================================================

def build_ood_dataset(
    ID_data,
    OOD_data,
    test_size=0.3,
    random_state=42,
    n_bins=10,
    n_kmeans_clusters=10,
    n_tree_partitions=5,
    tree_max_depth=3,
    use_raw_features=False,
    meta_features=None,
    partition_fit_data='id',
    balance_strategy='oversample',
):
    '''
    Build training and test datasets for OOD detection.

    1. Split ID data into train / test
    2. Fit ensemble of partitions
    3. Transform ID test and OOD data into the derived representation
    4. (Optional) Append scaled raw features to the representation
    5. Create labeled dataset (0 = ID, 1 = OOD)
    6. Scale the final representation

    Parameters
    ----------
    use_raw_features : bool
        If True, append the original (scaled) features to the
        partition-based representation.  The raw-feature scaler is fit
        on ID-train data only, so test / OOD data are scaled with the
        same statistics.
    partition_fit_data : {'id', 'id+ood'}
        Defines which data are used to fit partitioning schemes:
        - 'id': use only ID train split
        - 'id+ood': use ID train + OOD train splits

    balance_strategy : {'none', 'oversample', 'undersample'}
        Optional class balancing strategy applied on the train split only.

    Returns
    -------
    X_train, X_test, y_train, y_test, model, repr_scaler
    '''
    id_values = ID_data.values if isinstance(ID_data, pd.DataFrame) else ID_data
    ood_values = OOD_data.values if isinstance(OOD_data, pd.DataFrame) else OOD_data

    X_train_id, X_test_id = train_test_split(
        id_values, test_size=test_size, random_state=random_state
    )
    X_train_ood, X_test_ood = train_test_split(
        ood_values, test_size=test_size, random_state=random_state
    )

    return _build_ood_dataset_from_splits(
        X_train_id=X_train_id,
        X_test_id=X_test_id,
        X_train_ood=X_train_ood,
        X_test_ood=X_test_ood,
        random_state=random_state,
        n_bins=n_bins,
        n_kmeans_clusters=n_kmeans_clusters,
        n_tree_partitions=n_tree_partitions,
        tree_max_depth=tree_max_depth,
        use_raw_features=use_raw_features,
        meta_features=meta_features,
        partition_fit_data=partition_fit_data,
        balance_strategy=balance_strategy,
    )


# =====================================================================
# Classification
# =====================================================================

def classification_metrics(y_test, y_pred, y_proba):
    '''Compute classification metrics.'''
    roc_auc = round(sk_metrics.roc_auc_score(y_test, y_proba), 4)
    pr_auc = round(sk_metrics.average_precision_score(y_test, y_proba), 4)
    accuracy = round(sk_metrics.accuracy_score(y_test, y_pred), 4)
    precision = round(sk_metrics.precision_score(y_test, y_pred), 4)
    recall = round(sk_metrics.recall_score(y_test, y_pred), 4)
    f1 = round(sk_metrics.f1_score(y_test, y_pred), 4)

    return [roc_auc, pr_auc, accuracy, precision, recall, f1]


def train_ood_classifiers(
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_name='',
    include_mlp=True,
    random_state=42,
):
    '''
    Train Logistic Regression and Random Forest on partition-based
    representation and return results DataFrame and fitted classifiers.
    '''
    results_df = pd.DataFrame(
        columns=['roc-auc', 'pr-auc', 'accuracy', 'precision', 'recall', 'f1-score']
    )

    classifiers = {}

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        random_state=random_state,
        class_weight='balanced'
    )
    lr.fit(X_train, y_train)
    classifiers['Logistic Regression'] = lr

    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = lr.predict(X_test)
    results_df.loc['Logistic Regression'] = classification_metrics(
        y_test, y_pred_lr, y_proba_lr
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        criterion='log_loss',
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    classifiers['Random Forest'] = rf

    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    results_df.loc['Random Forest'] = classification_metrics(
        y_test, y_pred_rf, y_proba_rf
    )

    if include_mlp:
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=random_state,
        )
        mlp.fit(X_train, y_train)
        classifiers['MLP'] = mlp

        y_proba_mlp = mlp.predict_proba(X_test)[:, 1]
        y_pred_mlp = mlp.predict(X_test)
        results_df.loc['MLP'] = classification_metrics(
            y_test, y_pred_mlp, y_proba_mlp
        )

    return results_df, classifiers


def cross_validate_ood_classifiers(
    ID_data,
    OOD_data,
    n_splits=5,
    random_state=42,
    n_bins=10,
    n_kmeans_clusters=10,
    n_tree_partitions=5,
    tree_max_depth=3,
    use_raw_features=False,
    meta_features=None,
    partition_fit_data='id',
    balance_strategy='oversample',
    include_mlp=True,
):
    '''
    Cross-validate partition-based OOD classifiers with class balancing.
    '''
    id_values = ID_data.values if isinstance(ID_data, pd.DataFrame) else ID_data
    ood_values = OOD_data.values if isinstance(OOD_data, pd.DataFrame) else OOD_data

    kf_id = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf_ood = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows = []
    per_model_scores = {}

    for fold_idx, ((id_train_idx, id_test_idx), (ood_train_idx, ood_test_idx)) in enumerate(
        zip(kf_id.split(id_values), kf_ood.split(ood_values)),
        start=1
    ):
        X_train_id, X_test_id = id_values[id_train_idx], id_values[id_test_idx]
        X_train_ood, X_test_ood = ood_values[ood_train_idx], ood_values[ood_test_idx]

        X_train, X_test, y_train, y_test, _, _ = _build_ood_dataset_from_splits(
            X_train_id=X_train_id,
            X_test_id=X_test_id,
            X_train_ood=X_train_ood,
            X_test_ood=X_test_ood,
            random_state=random_state + fold_idx,
            n_bins=n_bins,
            n_kmeans_clusters=n_kmeans_clusters,
            n_tree_partitions=n_tree_partitions,
            tree_max_depth=tree_max_depth,
            use_raw_features=use_raw_features,
            meta_features=meta_features,
            partition_fit_data=partition_fit_data,
            balance_strategy=balance_strategy,
        )

        fold_metrics_df, classifiers = train_ood_classifiers(
            X_train, X_test, y_train, y_test,
            include_mlp=include_mlp,
            random_state=random_state + fold_idx,
        )

        for model_name, row in fold_metrics_df.iterrows():
            fold_rows.append({
                'fold': fold_idx,
                'model': model_name,
                'roc_auc': float(row['roc-auc']),
                'pr_auc': float(row['pr-auc']),
                'accuracy': float(row['accuracy']),
                'precision': float(row['precision']),
                'recall': float(row['recall']),
                'f1': float(row['f1-score']),
            })

            y_proba = classifiers[model_name].predict_proba(X_test)[:, 1]
            per_model_scores.setdefault(model_name, []).append(
                pd.DataFrame({'y_true': y_test, 'y_score': y_proba, 'fold': fold_idx})
            )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby('model', as_index=False)
        .agg({
            'roc_auc': ['mean', 'std'],
            'pr_auc': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std'],
        })
    )
    summary.columns = ['_'.join([c for c in col if c]) for col in summary.columns.values]
    summary = summary.rename(columns={'model_': 'model'})

    cv_scores = {k: pd.concat(v, ignore_index=True) for k, v in per_model_scores.items()}
    return fold_metrics, summary, cv_scores


# =====================================================================
# TPR at FPR
# =====================================================================

def tpr_at_fpr(classifiers, X_test, y_test, fpr_levels=None):
    '''Calculate TPR at given FPR levels for all classifiers.'''
    if fpr_levels is None:
        fpr_levels = [0.01, 0.05, 0.1]

    results = {}
    for name, clf in classifiers.items():
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = sk_metrics.roc_curve(y_test, y_proba)

        tpr_values = []
        for target_fpr in fpr_levels:
            idx = np.searchsorted(fpr, target_fpr, side='right') - 1
            tpr_values.append(round(tpr[max(0, idx)], 4))

        results[name] = dict(zip(fpr_levels, tpr_values))

    return pd.DataFrame(results).T


# =====================================================================
# Visualizations
# =====================================================================

def plot_roc_curves(classifiers, X_test, y_test, dataset_name=''):
    '''Plot ROC curves for all classifiers.'''
    plt.figure(figsize=(8, 6))

    for name, clf in classifiers.items():
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = sk_metrics.roc_curve(y_test, y_proba)
        auc = sk_metrics.roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name}, AUC: {auc:.4f}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.plot([0, 1], [0.9, 0.9], 'k--', label='90% TPR', color='tab:brown')

    plt.title(f'ROC Curves | {dataset_name}', fontweight='bold')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pr_curves(classifiers, X_test, y_test, dataset_name=''):
    '''Plot Precision-Recall curves for all classifiers.'''
    plt.figure(figsize=(8, 6))

    for name, clf in classifiers.items():
        y_proba = clf.predict_proba(X_test)[:, 1]
        precision, recall, _ = sk_metrics.precision_recall_curve(y_test, y_proba)
        ap = sk_metrics.average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, linewidth=2, label=f'{name}, AP: {ap:.4f}')

    plt.title(f'Precision-Recall Curves | {dataset_name}', fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_score_distributions(classifiers, X_test, y_test, dataset_name=''):
    '''Plot OOD score distributions for ID vs OOD.'''
    for name, clf in classifiers.items():
        y_proba = clf.predict_proba(X_test)[:, 1]
        id_scores = y_proba[y_test == 0]
        ood_scores = y_proba[y_test == 1]

        plt.figure(figsize=(10, 5))
        plt.hist(id_scores, bins=50, alpha=0.7, label='ID', density=True)
        plt.hist(ood_scores, bins=50, alpha=0.7, label='OOD', density=True)
        plt.xlabel('OOD Score')
        plt.ylabel('Density')
        plt.title(f'OOD Score Distribution | {name} | {dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_confusion_matrices(classifiers, X_test, y_test, dataset_name=''):
    '''Plot confusion matrices for all classifiers.'''
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        cm = sk_metrics.confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ID', 'OOD'], yticklabels=['ID', 'OOD']
        )
        plt.title(f'Confusion Matrix | {name} | {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()


def plot_cv_metric_distributions(fold_metrics, dataset_name=''):
    '''Visualize metric distributions across CV folds.'''
    metrics_to_plot = ['roc_auc', 'pr_auc', 'f1']
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=fold_metrics, x='model', y=metric)
        sns.stripplot(
            data=fold_metrics, x='model', y=metric,
            color='black', alpha=0.5, size=4
        )
        plt.title(f'CV {metric.upper()} by model | {dataset_name}')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.show()


def plot_cv_roc_curves(cv_scores, dataset_name=''):
    '''Plot mean ROC curve over folds for each model.'''
    plt.figure(figsize=(8, 6))
    base_grid = np.linspace(0, 1, 200)

    for model_name, score_df in cv_scores.items():
        curves = []
        for fold in sorted(score_df['fold'].unique()):
            chunk = score_df[score_df['fold'] == fold]
            fpr, tpr, _ = sk_metrics.roc_curve(chunk['y_true'], chunk['y_score'])
            tpr_interp = np.interp(base_grid, fpr, tpr)
            tpr_interp[0] = 0.0
            curves.append(tpr_interp)

        mean_tpr = np.mean(curves, axis=0)
        std_tpr = np.std(curves, axis=0)
        auc_mean = sk_metrics.auc(base_grid, mean_tpr)
        plt.plot(base_grid, mean_tpr, linewidth=2, label=f'{model_name}, mean AUC: {auc_mean:.4f}')
        plt.fill_between(
            base_grid,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.15
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.title(f'CV Mean ROC Curves | {dataset_name}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================================
# Full Pipeline
# =====================================================================

def full_pipeline(
    ID_data,
    OOD_data,
    dataset_name='',
    test_size=0.3,
    n_bins=10,
    n_kmeans_clusters=10,
    n_tree_partitions=5,
    tree_max_depth=3,
    random_state=42,
    use_raw_features=False,
    meta_features=None,
    partition_fit_data='id',
    balance_strategy='oversample',
    cv_folds=5,
    include_mlp=True,
):
    '''
    Full pipeline for partition-based OOD detection:
    1. Build partition representation
    2. Train classifiers (single split or CV)
    3. Evaluate quality (metrics + visualizations)
    '''
    print("=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)
    describe_dataset(ID_data, OOD_data)

    # ----- Build representation -----
    print("\n" + "=" * 70)
    print("BUILDING PARTITION REPRESENTATION")
    print("=" * 70)
    effective_mf = meta_features if meta_features is not None else DEFAULT_META_FEATURES
    print(f"Use raw features: {use_raw_features}")
    print(f"Partition fit data: {partition_fit_data}")
    print(f"Balance strategy: {balance_strategy}")
    print(f"CV folds: {cv_folds}")
    print(f"Meta features ({len(effective_mf)}): {effective_mf}")

    if cv_folds and cv_folds > 1:
        fold_metrics, summary_df, cv_scores = cross_validate_ood_classifiers(
            ID_data=ID_data,
            OOD_data=OOD_data,
            n_splits=cv_folds,
            random_state=random_state,
            n_bins=n_bins,
            n_kmeans_clusters=n_kmeans_clusters,
            n_tree_partitions=n_tree_partitions,
            tree_max_depth=tree_max_depth,
            use_raw_features=use_raw_features,
            meta_features=meta_features,
            partition_fit_data=partition_fit_data,
            balance_strategy=balance_strategy,
            include_mlp=include_mlp,
        )

        print("\n" + "=" * 70)
        print("CV CLASSIFICATION RESULTS (MEAN ± STD)")
        print("=" * 70)
        print(summary_df)

        print("\n" + "=" * 70)
        print("CV METRIC DISTRIBUTIONS")
        print("=" * 70)
        plot_cv_metric_distributions(fold_metrics, dataset_name)

        print("\n" + "=" * 70)
        print("CV MEAN ROC CURVES")
        print("=" * 70)
        plot_cv_roc_curves(cv_scores, dataset_name)

        legacy_df = summary_df[['model', 'roc_auc_mean', 'pr_auc_mean', 'accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']].copy()
        legacy_df = legacy_df.rename(columns={
            'roc_auc_mean': 'roc-auc',
            'pr_auc_mean': 'pr-auc',
            'accuracy_mean': 'accuracy',
            'precision_mean': 'precision',
            'recall_mean': 'recall',
            'f1_mean': 'f1-score',
        }).set_index('model')

        return {
            'cv_fold_metrics': fold_metrics,
            'cv_summary_df': summary_df,
            'cv_scores': cv_scores,
            'results_df': legacy_df,
        }

    X_train, X_test, y_train, y_test, model, repr_scaler = build_ood_dataset(
        ID_data, OOD_data,
        test_size=test_size,
        random_state=random_state,
        n_bins=n_bins,
        n_kmeans_clusters=n_kmeans_clusters,
        n_tree_partitions=n_tree_partitions,
        tree_max_depth=tree_max_depth,
        use_raw_features=use_raw_features,
        meta_features=meta_features,
        partition_fit_data=partition_fit_data,
        balance_strategy=balance_strategy,
    )

    print(f"Number of partitioning schemes: {len(model.schemes)}")
    print(f"Representation dimension:       {X_train.shape[1]}")
    print(f"Train set: {X_train.shape[0]} samples "
          f"(ID: {int((y_train == 0).sum())}, OOD: {int((y_train == 1).sum())})")
    print(f"Test set:  {X_test.shape[0]} samples "
          f"(ID: {int((y_test == 0).sum())}, OOD: {int((y_test == 1).sum())})")

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    results_df, classifiers = train_ood_classifiers(
        X_train, X_test, y_train, y_test, dataset_name,
        include_mlp=include_mlp, random_state=random_state
    )
    print(results_df)

    print("\n" + "=" * 70)
    print("TPR @ FPR")
    print("=" * 70)
    tpr_fpr_df = tpr_at_fpr(classifiers, X_test, y_test)
    print(tpr_fpr_df)

    print("\n" + "=" * 70)
    print("ROC CURVES")
    print("=" * 70)
    plot_roc_curves(classifiers, X_test, y_test, dataset_name)

    print("\n" + "=" * 70)
    print("PRECISION-RECALL CURVES")
    print("=" * 70)
    plot_pr_curves(classifiers, X_test, y_test, dataset_name)

    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS")
    print("=" * 70)
    plot_score_distributions(classifiers, X_test, y_test, dataset_name)

    print("\n" + "=" * 70)
    print("CONFUSION MATRICES")
    print("=" * 70)
    plot_confusion_matrices(classifiers, X_test, y_test, dataset_name)

    return {
        'results_df': results_df,
        'classifiers': classifiers,
        'model': model,
        'repr_scaler': repr_scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }
