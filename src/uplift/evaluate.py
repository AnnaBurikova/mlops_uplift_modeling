import torch 
import pandas as pd 
import numpy as np


def predict_slearner_uplift(model, X, treatment_col='is_treatment'): 
    if treatment_col in X.columns:
        X = X.drop(columns=[treatment_col], axis=1)
    return model.predict_uplift(X)

def predict_tlearner_uplift(model, X, treatment_col='is_treatment'): 
    if treatment_col in X.columns:
        X = X.drop(columns=['is_treatment'], axis=1)
    return model.predict_uplift(X)

def predict_tarnet_uplift(model, X, treatment_col='is_treatment'): 
    if treatment_col in X.columns:
        X = X.drop(columns=[treatment_col], axis=1)

    X = torch.tensor(X.values, dtype=torch.float32)
    X = torch.utils.data.TensorDataset(X)
    X = torch.utils.data.DataLoader(X, batch_size=128, shuffle=False)
    model.eval()
    uplifts = []
    for (x_batch,) in X: 
        uplifts.append(model.predict_uplift(x_batch).detach().cpu().numpy())
    return np.concatenate(uplifts)
