import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple

class DataSplitter:
    def __init__(self, train_part: float, val_part: float):
        self.train_part = train_part
        self.val_part = val_part
        self.random_state = 42

    def split_and_prepare(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:   
        if data.size == 0:
            print("Data size is 0. Nothing to prepare")
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        if not 'patient_id' in data.columns:
            data['patient_id'] = pd.Series()
        data = self.__fill_empty_patients_id(data)
        data.sort_values(by=['img_path'], inplace=True)

        return (self.__prepare(x) for x in self.__split(data))
    
    def __split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_part = self.train_part
        val_part = self.val_part
        test_part = 1.0 - train_part - val_part

        if test_part == 1.0:
            return data.iloc[:0,:].copy(), data.iloc[:0,:].copy(), data
        elif val_part == 1.0:
            return data.iloc[:0,:].copy(), data, data.iloc[:0,:].copy()
        elif train_part == 1.0:
            return data, data.iloc[:0,:].copy(), data.iloc[:0,:].copy()

        X = data
        y = data[['class']]
        groups = data[['patient_id']]
        train_idx, temp_idx = self.__split_with_boundary_awareness(train_part, X, y, groups)

        X_temp = data.iloc[temp_idx]
        y_temp = X_temp[['class']]
        groups_temp = X_temp[['patient_id']]

        rel_test_part = test_part / (test_part + val_part)
        rel_val_part = 1.0 - rel_test_part
        val_idx, test_idx = self.__split_with_boundary_awareness(rel_val_part, X_temp, y_temp, groups_temp)

        print(f"Data of size {data.shape[0]} split to sizes: \n train_size={len(train_idx)} \n validation_size={len(val_idx)} \n test_size={len(test_idx)}")

        return data.iloc[train_idx], X_temp.iloc[val_idx], X_temp.iloc[test_idx]

    def __split_with_boundary_awareness(self, train_size: float, X: pd.DataFrame, y, groups):
        if train_size == 1.0:
            return range(X.shape[0]), []

        if train_size == 0.0:
            return [], range(X.shape[0])
        
        gss = GroupShuffleSplit(train_size=train_size, random_state=self.random_state, n_splits=1)
        return next(gss.split(X, y, groups))

    def __prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.__shuffle_data(data)
        data = self.__drop_uneccessary_data(data)
        return data

    def __fill_empty_patients_id(self, df: pd.DataFrame) -> pd.DataFrame:
        update_df = pd.DataFrame(np.arange(len(df), 2*len(df)), columns=['patient_id'])
        df.update(update_df, overwrite=False)
        df['patient_id'] = df['patient_id'].astype(dtype=int)
        return df

    def __shuffle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.iloc[np.random.permutation(len(data))]

    def __drop_uneccessary_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop("patient_id", axis='columns', errors='ignore')
