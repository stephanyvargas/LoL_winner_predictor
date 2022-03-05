import pandas as pd
import numpy as np

from utils import get_synergy
from sklearn.base import BaseEstimator, TransformerMixin


class SynergyFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        #get the synergy matrix data
        champions_won_percentage_imputed = pd.read_csv('champions_won_percentage_imputed.csv')
        champions_won_percentage_imputed.index = champions_won_percentage_imputed['index']
        champions_won_percentage_imputed.drop(['index'], axis=1, inplace=True)
        self.synergy_matrix = champions_won_percentage_imputed

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        #Get the synergy of the team's champions
        df_test = X.apply(lambda z: [get_synergy(x,str(y), self.synergy_matrix) for x in z for y in z if x != y], axis=1).apply(np.mean)
        return pd.DataFrame(df_test)


class RoleFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        #get the synergy matrix data
        champions_won_percentage_imputed = pd.read_csv('champions_won_percentage_imputed.csv')
        champions_won_percentage_imputed.index = champions_won_percentage_imputed['index']
        champions_won_percentage_imputed.drop(['index'], axis=1, inplace=True)
        self.synergy_matrix = champions_won_percentage_imputed

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        #Get the synergy of the team's champions
        df_test = X.apply(lambda z: [get_synergy(x,str(y), self.synergy_matrix) for x in z for y in z if x != y], axis=1).apply(np.mean)
        return pd.DataFrame(df_test)
