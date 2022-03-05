import pandas as pd
import numpy as np

from utils import get_synergy, get_vs_rate
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
        df = X.apply(lambda z: [get_synergy(x,str(y), self.synergy_matrix) for x in z for y in z if x != y], axis=1).apply(np.mean)
        return pd.DataFrame(df)


class RoleFeature(BaseEstimator, TransformerMixin):

    def __init__(self, role):
        #get the role winrate champion vs champion DataFrame
        rate_champion_vs_champion = pd.read_csv('role_winrate_champ_vs_champ.csv',index_col=[0,1,2,3,4])
        self.rate_champion_vs_champion = rate_champion_vs_champion
        self.role = role

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        #Get the winrate of the same role champions
        df = X.apply(lambda z: get_vs_rate(z[0], self.role, z[1], self.rate_champion_vs_champion), axis=1)
        return pd.DataFrame(df)
