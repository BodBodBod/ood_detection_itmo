import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

_AE_CORE = Path(__file__).resolve().parent.parent / 'sample_wise_ae_ks'
if str(_AE_CORE) not in sys.path:
    sys.path.insert(0, str(_AE_CORE))

from pipeline import compute_reconstruction_errors

def tpr_by_fpr(
    model,
    ID_data: pd.DataFrame,
    OOD_data: pd.DataFrame,
    batch_size: int = 32,
    test_size: float = 0.3,
    significance_levels=[0.001, 0.01, 0.025, 0.05, 0.1]
):
    '''
    Методология:
    1. Задаем FPR (significance_levels)
    2. По нему определяем трешхолд значения ошибки реконструкции (MSE)
    3. По полученному трешхолду определяем к какому распределению принадлежит семпл: ID или OOD 
    4. Рассчитываем TPR (при заданном FPR)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_temp = train_test_split(ID_data, test_size=test_size, random_state=42)
    _, X_test_ID = train_test_split(X_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_test_ID_scaled = scaler.transform(X_test_ID)
    X_test_OOD_scaled = scaler.transform(OOD_data)

    test_ID_dataset = TensorDataset(torch.FloatTensor(X_test_ID_scaled))
    test_OOD_dataset = TensorDataset(torch.FloatTensor(X_test_OOD_scaled))

    test_ID_loader = DataLoader(test_ID_dataset, batch_size=batch_size, shuffle=False)
    test_OOD_loader = DataLoader(test_OOD_dataset, batch_size=batch_size, shuffle=False)
    
    ID_errors = compute_reconstruction_errors(model, test_ID_loader, device)
    OOD_errors = compute_reconstruction_errors(model, test_OOD_loader, device)

    tpr_arr = np.array([])
    threshold_arr = np.array([])
    for alpha in significance_levels:
        threshold = np.quantile(ID_errors, 1 - alpha)
        tpr = len(OOD_errors[OOD_errors >= threshold]) / len(OOD_errors)

        tpr_arr = np.append(tpr_arr, tpr)
        threshold_arr = np.append(threshold_arr, threshold)

    return [
        np.array(significance_levels),
        tpr_arr,
        threshold_arr
    ]

    
def plot_fpr_tpr(fpr, tpr):
    '''
    Строим зависимость TPR от FPR (моделируется функцией выше)
    '''
    plt.figure(figsize=(10, 5))

    plt.plot(fpr, tpr, marker='o', label='Actual Dependence', color='tab:blue')

    target_fpr = fpr[fpr <= 0.1]
    plt.plot(target_fpr, tpr[:len(target_fpr)], marker='o', label='TPR with FPR <= 0.1', color='tab:orange')

    plt.axhline(y=0.8, linestyle='--', color='black', label='Target Power')

    plt.ylim(-0.05, 1)
    #plt.xticks(fpr)

    # Добавляем подписи возле точек
    for x, y in zip(fpr, tpr):
        plt.annotate(
            f"{x}", 
            (x, y),
            textcoords="offset points",
            size=8,
            xytext=(0, 10),
            ha='center'
        )

    plt.grid(axis='y')
    
    plt.gca().tick_params(axis='x', which='both', bottom=False, labelbottom=False) 

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('TPR-FPR dependence (FPR value is annotated near each marker)')
    plt.legend()
    
    plt.show()