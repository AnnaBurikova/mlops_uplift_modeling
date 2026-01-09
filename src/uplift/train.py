from uplift.models import SLearner, TLearner, TarNet
from sklift.metrics import qini_auc_score
from loguru import logger 
import numpy as np

import torch 
from torch import optim, nn 
from torch.utils.data import DataLoader, TensorDataset



def train_s_learner(X_train, y_train, X_val, y_val, treatment_col='is_treatment', **catboost_params):
    model = SLearner(catboost_params, treatment_col)
    
    logger.info("Training S-Learner...")
    model.fit(X_train, y_train)
    logger.info("S-Learner trained successfully")

    val_preds = model.predict_uplift(X_val)
    val_score = qini_auc_score(y_val, val_preds, X_val[treatment_col])

    train_preds = model.predict_uplift(X_train)
    train_score = qini_auc_score(y_train, train_preds, X_train[treatment_col])

    logger.info(f"S-Learner Train Qini AUC: {train_score}")
    logger.info(f"S-Learner Qini AUC: {val_score}")

    return model 

def train_t_learner(X_train, y_train, X_val, y_val, catboost_params1:dict, catboost_params2:dict, treatment_col='is_treatment'):
    model = TLearner(catboost_params1, catboost_params2, treatment_col)
    
    logger.info("Training T-Learner...")
    model.fit(X_train, y_train)
    logger.info("T-Learner trained successfully")
    val_preds = model.predict_uplift(X_val)
    val_score = qini_auc_score(y_val, val_preds, X_val[treatment_col])

    train_preds = model.predict_uplift(X_train)
    train_score = qini_auc_score(y_train, train_preds, X_train[treatment_col])
    logger.info(f"T-Learner Train Qini AUC: {train_score}")
    logger.info(f"T-Learner Qini AUC: {val_score}")

    return model 


def train_tarnet(X_train, y_train, X_val, y_val, input_dim, hidden_dim, features_dim, learning_rate, epochs, batch_size, weight_decay, momentum, scheduler):
    model = TarNet(input_dim, hidden_dim, features_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler['step_size'], gamma=scheduler['gamma'])

    train_dataset = TensorDataset(
        torch.tensor(X_train.drop(columns=['is_treatment'], axis=1).values),
        torch.tensor(X_train['is_treatment'].values),
        torch.tensor(y_train.values)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.drop(columns=['is_treatment'], axis=1).values),
        torch.tensor(X_val['is_treatment'].values),
        torch.tensor(y_val.values)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_f = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, t, y in train_loader:
            treat_preds, contr_preds = model(x.float())
            treat_preds = treat_preds[torch.nonzero(t).view(-1), :]
            contr_preds = contr_preds[torch.nonzero(1-t).view(-1), :]
            loss = loss_f(treat_preds, y[torch.nonzero(t).view(-1)]) + loss_f(contr_preds, y[torch.nonzero(1-t).view(-1)])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

        val_loss = 0
        model.eval()
        for x, t, y in val_loader:
            treat_preds, contr_preds = model(x.float())
            treat_preds = treat_preds[torch.nonzero(t).view(-1), :]
            contr_preds = contr_preds[torch.nonzero(1-t).view(-1), :]
            loss = loss_f(treat_preds, y[torch.nonzero(t).view(-1)]) + loss_f(contr_preds, y[torch.nonzero(1-t).view(-1)])
            val_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss/len(val_loader)}")

        scheduler.step()

    train_y = [] 
    train_t = []
    train_uplift = []
    for x, t, y in train_loader: 
        treat_preds, contr_preds = model(x.float()) 
        train_y.append(y.detach().cpu().numpy())
        train_t.append(t.detach().cpu().numpy())
        train_uplift.append((treat_preds[:, 1] - contr_preds[:, 1]).detach().cpu().numpy())

    val_y = []
    val_t = [] 
    val_uplift = []
    for x, t, y in val_loader: 
        treat_preds, contr_preds = model(x.float()) 
        val_y.append(y.detach().cpu().numpy())
        val_t.append(t.detach().cpu().numpy())
        val_uplift.append((treat_preds[:, 1] - contr_preds[:, 1]).detach().cpu().numpy())

    train_y = np.concatenate(train_y)
    train_t = np.concatenate(train_t)
    train_uplift = np.concatenate(train_uplift)
    val_y = np.concatenate(val_y)
    val_t = np.concatenate(val_t)
    val_uplift = np.concatenate(val_uplift)

    train_score = qini_auc_score(train_y, train_uplift, train_t)
    val_score = qini_auc_score(val_y, val_uplift, val_t)

    logger.info(f"TarNet Train Qini AUC: {train_score}")
    logger.info(f"TarNet Val Qini AUC: {val_score}")

    return model 



