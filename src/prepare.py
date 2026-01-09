import pandas as pd
import numpy as np 
import yaml 
from sklearn.model_selection import train_test_split



if __name__ == '__main__': 
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    data = pd.read_csv('./data/raw/dataset.csv')

    ## Fetch params from params.yaml file 
    seed = params['data']['random_state']
    test_size = params['data']['test_size']
    val_size = params['data']['val_size']

    treatment=data[data['treatment_group']=='treatment'] 
    control=data[data['treatment_group']=='control']

    treatment=treatment.drop('treatment_group', axis=1)
    treatment['is_treatment'] = 1

    control=control.drop('treatment_group', axis=1)
    control['is_treatment'] = 0

    treatment, treatment_test = train_test_split(treatment, test_size=test_size, stratify=treatment['conversion'], random_state=seed)
    control, control_test = train_test_split(control, test_size=test_size, stratify=control['conversion'], random_state=seed)

    treatment_train, treatment_val = train_test_split(treatment, test_size=val_size, stratify=treatment['conversion'], random_state=seed)
    control_train, control_val = train_test_split(control, test_size=val_size, stratify=control['conversion'],random_state=seed)


    train, val, test = \
        pd.concat([treatment_train, control_train]),\
        pd.concat([treatment_val, control_val]),\
        pd.concat([treatment_test, control_test])

    train.to_csv('./data/processed/train.csv', index=False)
    val.to_csv('./data/processed/val.csv', index=False)
    test.to_csv('./data/processed/test.csv', index=False)