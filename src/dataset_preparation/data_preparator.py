import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from typing import Tuple, List

def prepare_data(data: pd.DataFrame, train_part: float, val_part: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data.size == 0:
        print("Data size is 0. Nothing to prepare")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    if not 'patient_id' in data.columns:
        data['patient_id'] = pd.Series()
    data = __fill_empty_patients_id(data)

    data.sort_values(by=['img_path'], inplace=True)

    return (__do_single_set_pipe(x) for x in __train_val_test_split(data, train_part=train_part, val_part=val_part, random_state=42))
    
def __train_val_test_split(data: pd.DataFrame, train_part: float, val_part: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_part = 1.0 - train_part - val_part

    X = data
    y = data[['class']]
    groups = data[['patient_id']]

    gss = GroupShuffleSplit(train_size=train_part, random_state=random_state, n_splits=1)
    train_idx, temp_idx = next(gss.split(X, y, groups))

    X_temp = data.iloc[temp_idx]
    y_temp = X_temp[['class']]
    groups_temp = X_temp[['patient_id']]

    rel_test_part = test_part / (test_part + val_part)
    rel_val_part = 1.0 - rel_test_part
    gss = GroupShuffleSplit(train_size=rel_val_part, random_state=random_state, n_splits=1)
    val_idx, test_idx = next(gss.split(X_temp, y_temp, groups_temp))

    print(f"Data split to sizes: \n train_size={len(train_idx)} \n validation_size={len(val_idx)} \n test_size={len(test_idx)}")

    return data.iloc[train_idx], data.iloc[val_idx], data.iloc[test_idx]

def __do_single_set_pipe(data: pd.DataFrame) -> pd.DataFrame:
    data = __shuffle_data(data)
    data = __drop_uneccessary_data(data)
    return data

def __fill_empty_patients_id(df: pd.DataFrame) -> pd.DataFrame:
    update_df = pd.DataFrame(np.arange(len(df), 2*len(df)), columns=['patient_id'])
    df.update(update_df, overwrite=False)
    df['patient_id'] = df['patient_id'].astype(dtype=int)
    return df

def __shuffle_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.iloc[np.random.permutation(len(data))]

def __drop_uneccessary_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop("patient_id", axis='columns', errors='ignore')
