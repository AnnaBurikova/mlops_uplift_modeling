import torch 
import pandas as pd 
import numpy as np

import warnings
from loguru import logger 

import yaml
import pickle
import json 

import matplotlib.pyplot as plt

from sklift.metrics import qini_auc_score, uplift_at_k, qini_curve, perfect_qini_curve 
from uplift.evaluate import predict_slearner_uplift, predict_tlearner_uplift, predict_tarnet_uplift
from uplift.models import TarNet

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__': 
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    seed = config['data']['random_state']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    test = pd.read_csv('./data/processed/test.csv') 
    train = pd.read_csv('./data/processed/train.csv')
    val = pd.read_csv('./data/processed/val.csv')

    X_test = test.drop(columns=['conversion', 'id'], axis=1)
    y_test = test['conversion']
    t_test = test['is_treatment']

    X_train = train.drop(columns=['conversion', 'id'], axis=1)
    y_train = train['conversion']
    t_train = train['is_treatment']

    X_val = val.drop(columns=['conversion', 'id'], axis=1)
    y_val = val['conversion']
    t_val = val['is_treatment']
    
    s_learner = pickle.load(open('models/s_learner.pkl', 'rb'))
    t_learner = pickle.load(open('models/t_learner.pkl', 'rb'))
    tarnet_learner_weights=torch.load(open('models/tarnet_learner.pth', 'rb'))
    tarnet_learner = TarNet(X_train.shape[1]-1, config['tarnet']['hidden_dim'], config['tarnet']['features_dim'])
    tarnet_learner.load_state_dict(tarnet_learner_weights)
    logger.info('Models loaded successfully...')
    models_names = ['S-Learner', 'T-Learner', 'TarNet']
    models = [s_learner, t_learner, tarnet_learner]
    model_funcs = [predict_slearner_uplift, predict_tlearner_uplift, predict_tarnet_uplift]

    models_name2func = {
        mn: func for mn, func in zip(models_names, model_funcs)
    }

    models_name2model = {
        mn: model for mn, model in zip(models_names, models)
    }


    results = []
    for mn in models_names:
        logger.info(f'Evaluating {mn}...')
        func = models_name2func[mn]
        test_uplift = func(models_name2model[mn], X_test, 'is_treatment')
        train_uplift = func(models_name2model[mn], X_train, 'is_treatment')
        val_uplift = func(models_name2model[mn], X_val, 'is_treatment')

        test_qini_auc=qini_auc_score(y_test, test_uplift, t_test)
        train_qini_auc=qini_auc_score(y_train, train_uplift, t_train)
        val_qini_auc=qini_auc_score(y_val, val_uplift, t_val)

        test_uplift_at_k=uplift_at_k(y_test, test_uplift, t_test, k=config['evaluate']['at_k'], strategy='overall')
        train_uplift_at_k=uplift_at_k(y_train, train_uplift, t_train, k=config['evaluate']['at_k'], strategy='overall')
        val_uplift_at_k=uplift_at_k(y_val, val_uplift, t_val, k=config['evaluate']['at_k'], strategy='overall')
    


        k1, test_qini = qini_curve(y_test, test_uplift, t_test)
        k2, test_best_qini = perfect_qini_curve(y_test, t_test)
        k3, test_random_qini = qini_curve(y_test, np.random.rand(len(y_test)), t_test)

        plt.plot(k1, test_qini, label='Model result')
        plt.plot(k2, test_best_qini, label='Best result')
        plt.plot(k3, test_random_qini, label='Random result')
        plt.title(f'Qini curve {mn}')
        plt.xlabel('Percentage of population')
        plt.ylabel('Qini coefficient')
        plt.legend()
        plt.savefig(f'reports/qini_curve_{mn}.png')
        plt.close()


        results.append({
            'model': mn,
            'test_qini_auc': test_qini_auc,
            'train_qini_auc': train_qini_auc,
            'val_qini_auc': val_qini_auc,
            f'test_uplift_at_k@{100*config["evaluate"]["at_k"]}': test_uplift_at_k,
            f'train_uplift_at_k@{100*config["evaluate"]["at_k"]}': train_uplift_at_k,
            f'val_uplift_at_k@{100*config["evaluate"]["at_k"]}': val_uplift_at_k
        })

    with open('reports/results.json', 'w') as f:
        json.dump(results, f)
    logger.info('Results saved successfully...')
    
