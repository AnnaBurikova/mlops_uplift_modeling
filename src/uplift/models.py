import torch 
from torch import nn 

import catboost as cb 


class SLearner: 
    def __init__(self, catboost_params:dict, treatment_col='is_treatment'): 
        self.model = cb.CatBoostClassifier(**catboost_params, verbose=False)
        self.treatment_col = treatment_col 

    def fit(self, X, y):
        self.model.fit(X, y, cat_features=[self.treatment_col]) 

    def predict_uplift(self, X): 
        X_treatment = X.copy()
        X_control = X.copy() 
        X_treatment[self.treatment_col] = 1
        X_control[self.treatment_col] = 0

        return self.model.predict_proba(X_treatment)[:,1] - self.model.predict_proba(X_control)[:, 1]


class TLearner: 
    def __init__(self, catboost_params1:dict, catboost_params2:dict, treatment_col='is_treatment'):
        self.tmodel = cb.CatBoostClassifier(**catboost_params1, verbose=False)
        self.cmodel = cb.CatBoostClassifier(**catboost_params2, verbose=False)
        self.treatment_col = treatment_col

    def fit(self, X, y):
        X_treatment = X[X[self.treatment_col]==1]
        X_control = X[X[self.treatment_col]==0]

        y_treatment = y[X[self.treatment_col]==1]
        y_control = y[X[self.treatment_col]==0]

        self.tmodel.fit(X_treatment.drop(columns=[self.treatment_col], axis=1), y_treatment)
        self.cmodel.fit(X_control.drop(columns=[self.treatment_col], axis=1), y_control)

    def predict_uplift(self, X):
        treat_preds = self.tmodel.predict_proba(X)[:,1]
        contr_preds = self.cmodel.predict_proba(X)[:,1]

        return treat_preds - contr_preds

class TarNet(nn.Module): 
    def __init__(self, input_dim, hidden_dim, features_dim):
        super(TarNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, features_dim)
        
        self.extractor = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

        self.treat_predictor = nn.Sequential(
            nn.Linear(features_dim, 2)
        )
        self.contr_predictor = nn.Sequential(
            nn.Linear(features_dim, 2)
        )

    def forward(self, x):
        x = self.extractor(x)
        treat_pred = self.treat_predictor(x)
        contr_pred = self.contr_predictor(x)
        return treat_pred, contr_pred

    def predict_uplift(self, X): 
        treat_preds, contr_preds = self.forward(X)
        return treat_preds[:, 1] - contr_preds[:, 1]

