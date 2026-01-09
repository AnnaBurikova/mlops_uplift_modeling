from uplift.train import train_s_learner, train_t_learner, train_tarnet
import pandas as pd
import numpy as np
import torch
import yaml
import pickle
import warnings



warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__': 
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    seed = config['data']['random_state']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train = pd.read_csv('./data/processed/train.csv')
    val = pd.read_csv('./data/processed/val.csv')
    test = pd.read_csv('./data/processed/test.csv')

    X_train = train.drop(columns=['conversion', 'id'], axis=1)
    y_train = train['conversion']
    X_val = val.drop(columns=['conversion', 'id'], axis=1)
    y_val = val['conversion']
    X_test = test.drop(columns=['conversion', 'id'], axis=1)
    y_test = test['conversion']

    s_learner = train_s_learner(X_train, y_train, X_val, y_val, **config['s_learner'], treatment_col='is_treatment')
    t_learner = train_t_learner(X_train, y_train, X_val, y_val, config['t_learner']['treatment_learner'], config['t_learner']['control_learner'], treatment_col='is_treatment')
    tarnet_learner = train_tarnet(X_train, y_train, X_val, y_val, len(X_train.columns)-1, config['tarnet']['hidden_dim'], config['tarnet']['features_dim'], config['tarnet']['learning_rate'], config['tarnet']['epochs'], config['tarnet']['batch_size'], config['tarnet']['weight_decay'], config['tarnet']['momentum'], config['tarnet']['scheduler'])
    
    torch.save(tarnet_learner.state_dict(), 'models/tarnet_learner.pth')
    with open('models/s_learner.pkl', 'wb') as f:
        pickle.dump(s_learner, f)

    with open('models/t_learner.pkl', 'wb') as f:
        pickle.dump(t_learner, f)