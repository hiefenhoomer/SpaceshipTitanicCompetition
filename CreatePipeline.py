import copy
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

set_config(transform_output='pandas')


class CreateExpenseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feat_dict, scalers_dict):
        print(f'Class: {self.__class__.__name__}.__init__')
        self.exp = 'exp'
        self.feat_dict = feat_dict
        self.scalers_dict = scalers_dict

    def fit(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.fit')
        # We've already fit the scalers; do nothing.
        return self

    def revert_columns(self, X, scaler_list, feat_list):
        for scaler, feat in zip(scaler_list, feat_list):
            X[feat] = scaler.inverse_transform(X[[feat]]).flatten()
            return X

    def transform(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.transform')
        X = X.copy()
        X_temp = pd.DataFrame()
        for scaler, feat in zip(self.scalers_dict[self.exp], self.feat_dict[self.exp]):
            # Have to flatten this two-dimensional array
            X_temp[f'inv_{feat}'] = scaler.inverse_transform(X[[feat]]).flatten()

        inv_feats = [f'inv_{feat}' for feat in self.feat_dict[self.exp]]
        X['Total'] = X_temp[inv_feats].sum(axis=1)
        prefix = 'prop'
        proportions = [f'{prefix}_{feat}' for feat in self.feat_dict[self.exp]]
        for prop_feat, inv_feat in zip(proportions, inv_feats):
            X[prop_feat] = X_temp[inv_feat].divide(X['Total'], axis=0).fillna(0)

        X = self.revert_columns(X, self.scalers_dict['num'], self.feat_dict['num'])
        X = self.revert_columns(X, self.scalers_dict['cat'], self.feat_dict['cat'])
        X = self.revert_columns(X, self.scalers_dict['bin'], self.feat_dict['bin'])

        self.feat_dict['prop_exp'] = proportions
        self.feat_dict['tot'] = ['Total']
        return X


class CustomKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feat_dict, n_neighbors=5):
        print(f'Class: {self.__class__.__name__}.__init__')
        self.feat_dict = feat_dict
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.cols = [col for cols in self.feat_dict.values() for col in cols]

    def fit(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.fit')
        # ["cab_Side", "num_Room", "cab_Deck"]
        missing_cols = [col for col in self.cols if col not in X.columns]
        if missing_cols:
            raise KeyError(f"Columns {missing_cols} are missing in the input DataFrame!!! WhaT THe FuCK!!!")
        self.imputer.fit(X[self.cols])
        return self

    def transform(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.transform')
        X = X.copy()
        X[self.cols] = self.imputer.transform(X[self.cols])
        return X


class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feat_dict, scaler_dict):
        print(f'At class {self.__class__.__name__}.__init__')
        self.feat_dict = feat_dict
        self.feat_dict_copy = copy.deepcopy(self.feat_dict)
        self.scaler_dict = scaler_dict
        self.scaler_dict_copy = copy.deepcopy(self.scaler_dict)
        for key in self.feat_dict.keys():
            self.scaler_dict[key] = [StandardScaler() for _ in feat_dict[key]]

    def fit(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.fit')
        for key in self.scaler_dict.keys():
            for scaler, col in zip(self.scaler_dict[key], self.feat_dict[key]):
                scaler.fit(X[[col]])
        return self

    def transform(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.transform')
        X = X.copy()
        for key in self.feat_dict_copy.keys():
            for scaler, col in zip(self.scaler_dict[key], self.feat_dict[key]):
                X[col] = scaler.transform(X[[col]])
        return X


class OrdinalBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, bin_feat):
        print(f'At class {self.__class__.__name__}.__init__')
        self.cols = bin_feat
        self.enc = OrdinalEncoder()

    def fit(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.fit')
        # Only need to fit to one column, the binary values are only one and zero
        self.enc.fit(X[[self.cols[0]]])
        return self

    def transform(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.transform')
        X = X.copy()
        for col in self.cols:
            print(col)
            X[col] = self.enc.transform(X[col])
        return X


class PandasDummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        print(f'At class {self.__class__.__name__}__init__')
        self.cols = cols
        self.true_vals = {}
        self.false_vals = {}

    def fit(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.fit')
        for col in self.cols:
            values = X[col].unique()
            self.false_vals[col] = values[0]
            self.true_vals[col] = values[1]
        return self

    def transform(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.transform')
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map({self.true_vals[col]: 1, self.false_vals[col]: 0})
        return X


class CustomOrdinalEnc(BaseEstimator, TransformerMixin):
    def __init__(self):
        print(f'At class {self.__class__.__name__}.__init__')
        # Returns the ordering of all categorical data. These are the decks in alphabetical order.
        # The sides in whichever order, it doesn't really matter since it's binary.
        # Home planets in ascending distance from the sun and destinations also in that same ordering.
        self.enc_categories = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', ''], ['Earth', 'Europa', 'Mars', ''],
                               ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22', '']]
        self.enc_cols = ['cab_Deck', 'HomePlanet', 'Destination']

        self.encoders = {col: OrdinalEncoder(categories=[cat]) for col, cat in zip(self.enc_cols, self.enc_categories)}

    def fit(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.fit')
        for col in self.enc_cols:
            # Fit valid data to the encoder.
            valid_data = X[[col]].dropna()
            self.encoders[col].fit(valid_data)
        return self

    def transform(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.transform')
        X = X.copy()
        for col in self.enc_cols:
            # Create boolean na mask to retain na values
            na_mask = X[col].isna()
            # Encode the columns including na values
            X[col] = self.encoders[col].transform(X[[col]].fillna(''))
            # Selects all row x column combinations where na is true and replaces with na.
            X.loc[na_mask, col] = np.NaN
        return X


class CabinSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, cabin_col='Cabin'):
        print(f'At class {self.__class__.__name__}.__init__')
        self.cabin_col = cabin_col

    def fit(self, X, y=None):
        print(f'Class: {self.__class__.__name__}.fit')
        return self

    def transform(self, X):
        print(f'Class: {self.__class__.__name__}.transform')
        X = X.copy()
        # Split the cabins by the delimiter '/'. Formatted like deck/room/side or A/22/S
        cabins = X[self.cabin_col].fillna('//').str.split('/', expand=True)
        X['cab_Deck'] = cabins[0].replace('', np.NaN)
        X['num_Room'] = cabins[1].replace('', np.NaN)
        X['cab_Side'] = cabins[2].replace('', np.NaN)
        X.drop(columns=[self.cabin_col], inplace=True)
        return X


def iterate_nested_dicts(nested_dict, _func):
    for key in nested_dict.keys():
        for item in nested_dict[key]:
            _func(item)


def create_pipeline(feat_dict):
    cabin_pipe = Pipeline(steps=[
        ('cabin_splitter', CabinSplitter())
    ])

    enc_pipe = Pipeline(steps=[
        ('enc', CustomOrdinalEnc()),
        ('bin', PandasDummyTransformer(feat_dict['bin']))
    ])

    scaler_dict = {}

    scaler_pipe = Pipeline(steps=[
        ('scaler', ScalerTransformer(feat_dict, scaler_dict))
    ])

    impute_pipe = Pipeline(steps=[
        ('imputer', CustomKNNImputer(feat_dict, n_neighbors=5))
    ])

    add_exp_feats_pipe = Pipeline(steps=[
        ('add_exp_feats', CreateExpenseFeatures(feat_dict, scaler_dict))
    ])

    full_pipe = Pipeline(steps=[
        ('cabin_split', cabin_pipe),
        ('enc_ord', enc_pipe),
        ('scaler', scaler_pipe),
        ('impute', impute_pipe),
        ('add_exp', add_exp_feats_pipe),
    ])

    return full_pipe
