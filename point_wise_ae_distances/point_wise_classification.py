import os
from sklearn import metrics

import numpy as np
from numpy.linalg import inv

import pandas as pd

from torch.utils.data import DataLoader

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from graphviz import Source

from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

_AE_CORE = Path(__file__).resolve().parent.parent / 'sample_wise_ae_ks'
if str(_AE_CORE) not in sys.path:
    sys.path.insert(0, str(_AE_CORE))

from pipeline import compute_reconstruction_errors_vec

def classification_metrics(estimator, X_test, y_test, threshold=None):
    y_proba = estimator.predict_proba(X_test)[:, 1]

    if threshold is None:
        y_pred = estimator.predict(X_test)
    else:
        y_pred = (y_proba >= threshold).astype(int)

    accuracy = round(metrics.accuracy_score(y_test, y_pred), 4)
    recall = round(metrics.recall_score(y_test, y_pred), 4)
    precision = round(metrics.precision_score(y_test, y_pred), 4)
    f1 = round(metrics.f1_score(y_test, y_pred), 4)
    log_loss = round(metrics.log_loss(y_test, y_proba), 4)
    roc_auc = round(metrics.roc_auc_score(y_test, y_proba), 4)

    return [roc_auc, accuracy, precision, recall, f1, log_loss]


def roc(estimator, X_test, y_test):
    pred_proba = estimator.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_proba[:, 1])

    return fpr, tpr, thresholds, metrics.roc_auc_score(y_test, pred_proba[:, 1])


def save_tree(save_path, filename, tree, feature_names):
    dot_data = export_graphviz(
                            tree, feature_names=feature_names,
                            out_file=None,
                            filled=True, rounded=True,
                            special_characters=True)

    graph = Source(dot_data)
    graph.render(f"{save_path}/{filename}", view=False, format='pdf')
    os.remove(f"{save_path}/{filename}")


def calculate_error_metrics(
    matrix,
    metrics=None,
    feature_stds=None,
    id_mean=None,
    id_cov=None,
    cov_reg=1e-6
):

    if metrics is None:
        metrics = [
            # базовые нормы
            "mse", "rmse", "mae", "l1", "l2", "linf",
            # статистические
            "mean", "std", "skew", "kurtosis",
            # структурные
            "entropy", "gini",
            # threshold-based
            "count_above_std", "count_above_2std", "count_above_3std",
            "mahalanobis",
            "log_likelihood"
        ]

    X = matrix
    n_samples, n_features = X.shape

    results = {}

    # ---------- базовые нормы ----------
    if "mse" in metrics:
        results["mse"] = np.mean(X**2, axis=1)
    if "rmse" in metrics:
        results["rmse"] = np.sqrt(np.mean(X**2, axis=1))
    if "mae" in metrics:
        results["mae"] = np.mean(np.abs(X), axis=1)
    if "l1" in metrics:
        results["l1"] = np.sum(np.abs(X), axis=1)
    if "l2" in metrics:
        results["l2"] = np.sqrt(np.sum(X**2, axis=1))
    if "linf" in metrics:
        results["linf"] = np.max(np.abs(X), axis=1)

    # ---------- статистические ----------
    if "mean" in metrics:
        results["mean"] = np.mean(X, axis=1)
    if "std" in metrics:
        results["std"] = np.std(X, axis=1)
    if "skew" in metrics:
        results["skew"] = stats.skew(X, axis=1)
    if "kurtosis" in metrics:
        results["kurtosis"] = stats.kurtosis(X, axis=1)

    # ---------- энтропия и Gini ----------
    if "entropy" in metrics or "gini" in metrics:
        absX = np.abs(X)
        absX_sum = absX.sum(axis=1, keepdims=True) + 1e-12
        p = absX / absX_sum  # распределение весов ошибок

    if "entropy" in metrics:
        results["entropy"] = -(p * np.log(p + 1e-12)).sum(axis=1)

    if "gini" in metrics:
        # Gini per sample
        sorted_p = np.sort(p, axis=1)
        n = X.shape[1]
        G = (np.sum((2 * np.arange(1, n+1) - n - 1) * sorted_p, axis=1)) / (n - 1)
        results["gini"] = G

    # ---------- threshold-based ----------
    if "count_above_std" in metrics:
        std_feat = X.std(axis=0)
        thr = 1 * (std_feat + 1e-12)
        results["count_above_std"] = (np.abs(X) > thr).sum(axis=1)

    if "count_above_2std" in metrics:
        std_feat = X.std(axis=0)
        thr = 2 * (std_feat + 1e-12)
        results["count_above_2std"] = (np.abs(X) > thr).sum(axis=1)

    if "count_above_3std" in metrics:
        std_feat = X.std(axis=0)
        thr = 3 * (std_feat + 1e-12)
        results["count_above_3std"] = (np.abs(X) > thr).sum(axis=1)

    # ---------- mahalanobis ----------
    if "mahalanobis" in metrics:
        if id_mean is None or id_cov is None:
            raise ValueError("id_mean and id_cov must be provided for Mahalanobis")
        
        cov_reg_matrix = id_cov + cov_reg * np.eye(n_features)
        inv_cov = inv(cov_reg_matrix)

        diffs = X - id_mean
        results["mahalanobis"] = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))

    # ---------- Log-likelihood per feature (Gaussian) ----------
    if "log_likelihood" in metrics:
        if feature_stds is None:
            raise ValueError("feature_stds required for log_likelihood")
        var = (feature_stds**2 + 1e-12)
        ll = -0.5 * (np.log(2*np.pi*var).sum() + ((X**2)/var).sum(axis=1))
        results["log_likelihood"] = ll

    # ---------- косинусное расстояние к среднему ID-вектору ----------
    if "cosine_to_id_mean" in metrics:
        if id_mean is None:
            raise ValueError("id_mean must be provided for cosine_to_id_mean")

        # вектор нормы id_mean
        id_norm = np.linalg.norm(id_mean) + 1e-12

        # нормы строк X
        X_norm = np.linalg.norm(X, axis=1) + 1e-12

        # скалярные произведения
        dot_prod = X @ id_mean

        # cos = (x·μ)/(||x|| ||μ||)
        cos_sim = dot_prod / (X_norm * id_norm)

        # расстояние = 1 - cos_sim
        results["cosine_to_id_mean"] = 1 - cos_sim
    
    out = np.column_stack([results[m] for m in metrics])
    return out


def classification_preprocess(res, metrics, OOD_size=None, ID_size=None, test_size=0.3, use_distance=True):
    ID_size = ID_size if ID_size else len(res[1])
    OOD_size = OOD_size if OOD_size else len(res[2])

    ID_test_data = res[1][:ID_size][0]
    OOD_test_data = res[2][:OOD_size][0]
    
    reconstruction_error_matrix_ID = compute_reconstruction_errors_vec(
        res[0], 
        DataLoader(ID_test_data, batch_size=32, shuffle=False), 
        res[3]
    )

    reconstruction_error_matrix_OOD = compute_reconstruction_errors_vec(
        res[0], 
        DataLoader(OOD_test_data, batch_size=32, shuffle=False), 
        res[3]
    )

    id_mean = np.mean(reconstruction_error_matrix_ID, axis=0)
    id_cov = np.cov(reconstruction_error_matrix_ID, rowvar=False)
    
    if use_distance:
        distance_OOD = calculate_error_metrics(reconstruction_error_matrix_OOD, metrics=metrics, id_mean=id_mean, id_cov=id_cov)
        distance_ID = calculate_error_metrics(reconstruction_error_matrix_ID, metrics=metrics, id_mean=id_mean, id_cov=id_cov)
    else:
        distance_OOD = reconstruction_error_matrix_OOD.copy()
        distance_ID = reconstruction_error_matrix_ID.copy()

    distance = np.concatenate([distance_OOD, distance_ID])

    labels = np.concatenate([np.ones(len(distance_OOD)), np.zeros(len(distance_ID))])
    
    # train test split
    scaler = StandardScaler()
    scaler.fit(distance_ID)

    distance_scaled = scaler.transform(distance)
    X_train, X_test, y_train, y_test = train_test_split(
        distance_scaled, 
        labels, 
        test_size=test_size, 
        random_state=42,
        stratify=labels
    )

    return X_train, X_test, y_train, y_test


def classification_preprocess_synthetic(res, metrics, OOD_size=None, ID_size=None, test_size=0.3, noise_mean=0.0, noise_std=1.0):
    ID_size = ID_size if ID_size else len(res[1])
    OOD_size = OOD_size if OOD_size else len(res[2])

    ID_test_data = res[1][:ID_size][0]
    OOD_test_data = res[2][:OOD_size][0]
    
    reconstruction_error_matrix_ID = compute_reconstruction_errors_vec(
        res[0], 
        DataLoader(ID_test_data, batch_size=32, shuffle=False), 
        res[3]
    )

    # distance & scaling ID 
    id_mean = np.mean(reconstruction_error_matrix_ID, axis=0)
    id_cov = np.cov(reconstruction_error_matrix_ID, rowvar=False)
    distance_ID = calculate_error_metrics(reconstruction_error_matrix_ID, metrics=metrics, id_mean=id_mean, id_cov=id_cov)
    
    scaler = StandardScaler()
    distance_ID_scaled = scaler.fit_transform(distance_ID)

    # разбить distance_ID на train и test
    X_train_ID, X_test_ID, y_train_ID, y_test_ID = train_test_split(
        distance_ID_scaled, 
        np.zeros(len(distance_ID)), 
        test_size=test_size, 
        random_state=42
    )

    # искусственно изменить (1 - test_size) долю записей reconstruction_error_matrix_ID сместив на случайный вектор
    cnt_ID_for_synthetic_transformation = int((1 - test_size) * len(reconstruction_error_matrix_ID))
    reconstruction_error_matrix_OOD_synthetic = reconstruction_error_matrix_ID[:cnt_ID_for_synthetic_transformation]
    distance_OOD_synthetic = calculate_error_metrics(reconstruction_error_matrix_OOD_synthetic, metrics=metrics, id_mean=id_mean, id_cov=id_cov)
    distance_OOD_synthetic_scaled = scaler.transform(distance_OOD_synthetic)
    
    # synthetic transformation
    noise = np.random.normal(
        loc=noise_mean,
        scale=noise_std,
        size=distance_OOD_synthetic_scaled.shape
    )

    distance_OOD_synthetic_scaled = distance_OOD_synthetic_scaled + noise

    # добавить test_size долю OOD данных в конец
    reconstruction_error_matrix_OOD = compute_reconstruction_errors_vec(
        res[0], 
        DataLoader(OOD_test_data, batch_size=32, shuffle=False), 
        res[3]
    )    
    cnt_OOD_for_test = int(test_size * len(reconstruction_error_matrix_ID)) # берем test_size долю от ID данных, чтобы использовать тот же test_size в train_test_split 
    reconstruction_error_matrix_OOD = reconstruction_error_matrix_OOD[:cnt_OOD_for_test]
 
    # distance & scaling
    distance_OOD_real = calculate_error_metrics(reconstruction_error_matrix_OOD, metrics=metrics, id_mean=id_mean, id_cov=id_cov)
    distance_OOD_real_scaled = scaler.transform(distance_OOD_real)

    distance_OOD_scaled = np.concatenate([distance_OOD_synthetic_scaled, distance_OOD_real_scaled])

    # разбить distance_OOD на train и test
    X_train_OOD, X_test_OOD, y_train_OOD, y_test_OOD = train_test_split(
        distance_OOD_scaled, 
        np.ones(len(distance_OOD_scaled)), 
        test_size=test_size, 
        random_state=42,
        shuffle=False
    )

    # финальный набор датасетов для обучения и тестов
    X_train = np.concatenate([X_train_ID, X_train_OOD])
    X_test = np.concatenate([X_test_ID, X_test_OOD])
    y_train = np.concatenate([y_train_ID, y_train_OOD])
    y_test = np.concatenate([y_test_ID, y_test_OOD])

    return X_train, X_test, y_train, y_test

def train_models(
    X_train, X_test, y_train, y_test, 
    models, 
    save_tree_flg=True, 
    feature_names=None,
    dataset_name='',
    plot_roc=True
):
    results_df = pd.DataFrame(columns=['roc-auc', 'accuracy', 'precision', 'recall', 'f1-score', 'log-loss'])
    if 'logreg' in models:
        logreg = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=42
        )
        logreg.fit(X_train, y_train)

        results_df.loc['Logistic Regression'] = classification_metrics(logreg, X_test, y_test)

    if 'forest' in models:
        forest = RandomForestClassifier(n_estimators=50, random_state=42, criterion='log_loss')
        forest.fit(X_train, y_train)

        results_df.loc['Random Forest'] = classification_metrics(forest, X_test, y_test)
    
    if 'tree' in models:
        tree = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='log_loss')
        tree.fit(X_train, y_train)

        results_df.loc['Decision Tree'] = classification_metrics(tree, X_test, y_test)
        if save_tree_flg:
            save_tree('images/decision_tree', dataset_name, tree, feature_names=feature_names)

    if plot_roc:
        plt.figure(figsize=(8,6))
        if 'logreg' in models:
            fpr, tpr, thresholds, auc = roc(logreg, X_test, y_test)

            plt.plot(
                fpr, tpr, 
                linewidth=2,
                label=f'Logistic Regression, AUC: {auc:.2f}'
            )

        if 'forest' in models:
            fpr, tpr, thresholds, auc = roc(forest, X_test, y_test)

            plt.plot(
                fpr, tpr, 
                linewidth=2,
                label=f'Random Forest, AUC: {auc:.2f}'
            )

        if 'tree' in models:
            fpr, tpr, thresholds, auc = roc(tree, X_test, y_test)

            plt.plot(
                fpr, tpr, 
                linewidth=2,
                label=f'Tree, AUC: {auc:.2f}'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.plot([0, 1], [0.9, 0.9], 'k--', label='90% TPR', color='tab:brown')

        plt.title(f'ROC Curves | {dataset_name}', fontweight='bold')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid()
        plt.legend()

        plt.show()

    return results_df


def print_corr_matrix(distance, labels):
    corr_matrix = np.corrcoef(np.column_stack([distance, labels]), rowvar=False)

    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Bank Data")
    plt.show()