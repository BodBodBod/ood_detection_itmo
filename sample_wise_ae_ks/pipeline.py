from model import Autoencoder

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm_notebook, tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from matplotlib import pyplot as plt
import seaborn as sns

from warnings import filterwarnings

filterwarnings('ignore')


def compute_reconstruction_errors(model, data_loader, device):
    '''
    MSE between input and output
    '''
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            x = x.to(device)
            reconstructed = model(x)
            
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)

def compute_reconstruction_errors_vec(model, data_loader, device):
    '''
    reconstruction errors vector between input and output
    '''
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            x = x.to(device)
            reconstructed = model(x)
            
            errors.extend((x - reconstructed).cpu().numpy())
    
    return np.array(errors)


def train_ae_with_early_stopping(model, train_loader, val_loader, epochs=1000, patience=20, lr=0.001):
    '''
    Autoencoder training
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            x = x.to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                    
                x = x.to(device)
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # best
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


def plot_reconstruction_errors(ID_errors, OOD_errors):
    '''
    Reconstruction error distribution: histogram and boxplot
    '''
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(ID_errors, bins=50, alpha=0.7, label='ID', density=True)
    plt.hist(OOD_errors, bins=50, alpha=0.7, label='OOD', density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([ID_errors, OOD_errors], labels=['ID', 'OOD'])
    plt.title('Boxplot of Reconstruction Errors')
    plt.ylabel('Reconstruction Error')

    plt.tight_layout()
    plt.show()


def pipeline(
    ID_data: pd.DataFrame,
    OOD_data: pd.DataFrame,
    latent_dim: int = 3,
    batch_size: int = 32,
    test_size: float = 0.3,
    hidden_dims: list = [64, 32]
):
    '''
    AE training and results visualization
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(ID_data.columns)

    X_train, X_temp = train_test_split(ID_data, test_size=test_size, random_state=42)
    X_val, X_test_ID = train_test_split(X_temp, test_size=0.5, random_state=42) # X for validation and X for comparasion with OOD

    print("="*70)
    print(f"DATA VOLUME")
    print("="*70)
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test ID: {X_test_ID.shape}")
    print(f"Test OOD: {OOD_data.shape}")


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_ID_scaled = scaler.transform(X_test_ID)
    X_test_OOD_scaled = scaler.transform(OOD_data)

    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))    
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled))
    test_ID_dataset = TensorDataset(torch.FloatTensor(X_test_ID_scaled))
    test_OOD_dataset = TensorDataset(torch.FloatTensor(X_test_OOD_scaled))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_ID_loader = DataLoader(test_ID_dataset, batch_size=batch_size, shuffle=False)
    test_OOD_loader = DataLoader(test_OOD_dataset, batch_size=batch_size, shuffle=False)


    print("\n" + "="*70)
    print(f"TRAINING")
    print("="*70)
    model = Autoencoder(input_dim, latent_dim, hidden_dims=hidden_dims)
    model, train_losses, val_losses = train_ae_with_early_stopping(
        model, train_loader, val_loader, epochs=1000, patience=30, lr=0.001
    )

    
    print("\n" + "="*70)
    print(f"LOSS FUNCTION")
    print("="*70)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True)
    plt.show()


    print("\n" + "="*70)
    print(f"RECONSTRUCTION ERRORS")
    print("="*70)

    ID_errors = compute_reconstruction_errors(model, test_ID_loader, device)
    OOD_errors = compute_reconstruction_errors(model, test_OOD_loader, device)

    print(f"Mean reconstruction error ID: {np.mean(ID_errors):.6f}")
    print(f"Mean reconstruction error OOD: {np.mean(OOD_errors):.6f}")

    plot_reconstruction_errors(ID_errors, OOD_errors)

    
    print("\n" + "="*70)
    print(f"KS-TEST")
    print("="*70)
    ks_statistic, p_value = stats.ks_2samp(ID_errors, OOD_errors)
    print(f"statistic: {ks_statistic:.6f}")
    print(f"p-value: {p_value:.6f}")

    return model, test_ID_dataset, test_OOD_dataset, device


def bootstrap_sample(dataset, sample_size):
    indices = torch.randint(0, len(dataset), (sample_size,))
    return Subset(dataset, indices)


def monte_carlo(
        model, test_ID_dataset, test_OOD_dataset, device,
        num_experiments=10_000, sample_size=1_000, alpha=0.05,
        loader_batch_size=32
):
    ''' 
    Monte Carlo process for evaluating FPR and TPR
    '''
    fp_counter = 0
    tp_counter = 0
    for _ in tqdm(range(num_experiments)):
        ID1 = DataLoader(bootstrap_sample(test_ID_dataset, sample_size), batch_size=loader_batch_size, shuffle=False)
        ID2 = DataLoader(bootstrap_sample(test_ID_dataset, sample_size), batch_size=loader_batch_size, shuffle=False)
        OOD = DataLoader(bootstrap_sample(test_OOD_dataset, sample_size), batch_size=loader_batch_size, shuffle=False)
        
        ID1_errors = compute_reconstruction_errors(model, ID1, device)
        ID2_errors = compute_reconstruction_errors(model, ID2, device)
        OOD_errors = compute_reconstruction_errors(model, OOD, device)

        p_value_fpr = stats.ks_2samp(ID1_errors, ID2_errors)[1]
        p_value_tpr = stats.ks_2samp(ID1_errors, OOD_errors)[1]
        
        if p_value_fpr <= alpha: 
            fp_counter += 1

        if p_value_tpr <= alpha: 
            tp_counter += 1

    fpr = fp_counter / num_experiments
    tpr = tp_counter / num_experiments

    return {
        'FPR': fpr,
        'FPR confint': proportion_confint(fp_counter, num_experiments, alpha=alpha, method='wilson'),
        'TPR': tpr,
        'TPR confint': proportion_confint(tp_counter, num_experiments, alpha=alpha, method='wilson')
    }


def make_id_ood_mixture(
        ID_data: torch.Tensor, 
        OOD_data: torch.Tensor,
        id_volume: float, 
        volume: int = 1_000
):
    '''
    Create mixture of ID and OOD data
    with ID share as id_volume parameter
    '''
    id_size = int(id_volume * volume)
    ood_size = volume - id_size
    
    ID_indices = torch.randint(0, len(ID_data), (id_size,))
    OOD_indices = torch.randint(0, len(OOD_data), (ood_size,))
    
    ID_sample = ID_data[ID_indices][0]
    OOD_sample = OOD_data[OOD_indices][0]

    mixture = torch.cat([ID_sample, OOD_sample], dim=0)
    
    return mixture


def make_id_ood_mixture_with_labels(
        ID_data: torch.Tensor, 
        OOD_data: torch.Tensor,
        id_volume: float, 
        volume: int = 1_000
):
    id_size = int(id_volume * volume)
    ood_size = volume - id_size

    # sampling with replacement
    ID_indices = torch.randint(0, ID_data.size(0), (id_size,), device=ID_data.device)
    OOD_indices = torch.randint(0, OOD_data.size(0), (ood_size,), device=OOD_data.device)

    ID_sample = ID_data[ID_indices]
    OOD_sample = OOD_data[OOD_indices]

    # labels
    ID_labels = torch.zeros((id_size, 1), device=ID_data.device)
    OOD_labels = torch.ones((ood_size, 1), device=OOD_data.device)

    # datasets with labels (ID / OOD)
    ID_sample = torch.cat([ID_sample, ID_labels], dim=1)
    OOD_sample = torch.cat([OOD_sample, OOD_labels], dim=1)

    mixture = torch.cat([ID_sample, OOD_sample], dim=0)

    return mixture


def tpr_on_mixtures(
    res,
    mixture_levels=np.concatenate((np.arange(0.05, 0.75, 0.1), np.arange(0.7, 1, 0.05))),
    N_experiments=5000,
    sample_size=1000,
    alpha=0.05,
):
    '''
    Monte Carlo process for TPR estimation in mixture sample case
    '''
    model = res[0]

    ID_test_data = res[1]
    OOD_test_data = res[2]

    device = res[3]

    ID_test_data_loader = DataLoader(ID_test_data[:sample_size][0], batch_size=32, shuffle=True)
    ID_reconstruction_errors = compute_reconstruction_errors(model, ID_test_data_loader, device)

    tp_data = []
    for level in mixture_levels:
        tp_counter = 0
        for _ in tqdm(range(N_experiments)):
            mixture_data = make_id_ood_mixture(ID_test_data, OOD_test_data, id_volume=level, volume=sample_size)
            mixture_data_loader = DataLoader(mixture_data, batch_size=32, shuffle=True)

            mixture_reconstruction_errors = compute_reconstruction_errors(model, mixture_data_loader, device)
            
            pvalue = stats.ks_2samp(
                ID_reconstruction_errors, 
                mixture_reconstruction_errors
            )[1]

            if pvalue <= alpha: 
                tp_counter += 1
        
        tpr = tp_counter / N_experiments
        confint = proportion_confint(count=tp_counter, nobs=N_experiments, alpha=alpha, method='wilson')
        tp_data.append((tpr, confint))

    return mixture_levels, tp_data


def plot_tpr_mixtures(mixture_levels, tp_data):
    '''
    plot dependency TPR(FPR) for each mixture level (share of ID data in sample)
    '''
    tpr = [item[0] for item in tp_data]
    confint_lower = [item[1][0] for item in tp_data]
    confint_upper = [item[1][1] for item in tp_data]

    plt.figure(figsize=(14,5))

    plt.plot(mixture_levels, tpr, marker='o', markersize=5, alpha=0.7, label='TPR')
    plt.fill_between(mixture_levels, confint_lower, confint_upper, alpha=0.8, color='tab:red', label='TPR Confidence Interval') 

    plt.xticks(mixture_levels)

    plt.grid()
    plt.legend()

    plt.title('Dependence of TPR by share of ID objects in test data')
    plt.ylabel('TPR (share of samples detected as OOD)')
    plt.xlabel('Share of ID Objects in Sample')

    plt.show()