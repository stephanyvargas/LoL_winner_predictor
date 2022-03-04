from utils import get_synergy, pair_wise_synergy
from sklearn.base import BaseEstimator, TransformerMixin


class SynergyFeature(BaseEstimator, TransformerMixin):

    def __init__(self, team_side, df_team_side):
        self.team_side = team_side
        self.df_team_side = df_team_side

    def fit(self, df_team_side, y=None):
        return self

    def transform(self, df_team_side, y=None):
        assert isinstance(df_team_side, pd.DataFrame)

        champions_won_percentage_imputed = pd.read_csv('champions_won_percentage_imputed.csv')
        get_synergy(x,y, df_synergy_matrix)
        pair_wise_synergy(df, df_synergy_matrix, team_side)

        #Get the synergy of the team's champions
        df_team = pair_wise_synergy(df_team_side, champions_won_percentage_imputed, champions_won_percentage_imputed)
        df_team['id'] = df_team.index
        df_teams_ML =pd.merge(df_teams_ML, df_team, on='id', how='inner')
        return df_teams_ML
