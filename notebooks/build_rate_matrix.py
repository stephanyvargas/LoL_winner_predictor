import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import SimpleImputer


"""Script to build the different matrices needed to get the features."""

def synergy_matrix(df_BLUE, df_RED):
    df_BLUE_lite_synergy = df_BLUE[['champion_id', 'win', 'game_id']]
    df_RED_lite_synergy = df_RED[['champion_id', 'win', 'game_id']]

    #Dosen't matter if it was the red or blue team so let's concat the data!
    df_result = pd.concat([df_BLUE_lite_synergy, df_RED_lite_synergy])

    #Include a column for the outcome
    df_result['outcome'] = df_result['win']*1

    #Reset the index to the game_id
    df_result.index = df_result.game_id

    # get the unique game_id and champion_id played
    game_id_unique = np.unique(df_result.game_id)
    champions_id_unique = np.unique(df_result.champion_id)

    #create two dataframes
    ##one to populate the number of games played by each champion pair
    champions_play_together = pd.DataFrame(np.zeros([len(champions_id_unique),
                                           len(champions_id_unique)]),
                                           columns=champions_id_unique,
                                           index=champions_id_unique)

    for champion_a in tqdm(champions_id_unique):
        champions_played_together_list = list(df_result.loc[df_result[df_result.champion_id == champion_a].index]['champion_id'])
        for champion_b in champions_played_together_list:
            if champion_a == champion_b:
                continue
            else:
                champions_play_together.loc[champion_a][champion_b] += 1

    ##one to populate te number of games won by each champion pair
    champions_won_together = pd.DataFrame(np.zeros([len(champions_id_unique),
                                                    len(champions_id_unique)]),
                                                    columns=champions_id_unique,
                                                    index=champions_id_unique)

    for champ_a in tqdm(champions_id_unique):
        champ_played_together_won_list = list(df_result.loc[df_result[(df_result.champion_id == champ_a) & (df_result.outcome == 1)].index]['champion_id'])
        for champ_b in champ_played_together_won_list:
            if champ_a == champ_b:
                continue
            else:
                champions_won_together.loc[champ_a][champ_b] += 1

    #get the win rate by dividing the number or wins over the total times played
    champions_won_percentage = champions_won_together.div(champions_play_together)

    #Impute the missing values (champions that never played together)
    impute_nan = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.5)
    champions_won_percentage_imputed = pd.DataFrame(impute_nan.fit(champions_won_percentage).transform(champions_won_percentage), columns=champions_id_unique, index=champions_id_unique)
    champions_won_percentage_imputed

    champions_won_percentage_imputed.to_csv('champions_won_percentage_imputed.csv', index_label='index')

    #Sanity check
    ##check that there is no division by zero
    '''np.isinf(champions_won_percentage_imputed).values.sum()'''

    ##check how many values have been imputed
    '''np.isnan(champions_won_percentage_imputed).values.sum()
    np.isnan(champions_won_percentage).values.sum()'''

    return champions_won_percentage_imputed


def Role_DataFrame(df_BLUE, df_RED):
    df_BLUE_RED = pd.merge(left=df_BLUE, right=df_RED, left_on= 'game_id', right_on= 'game_id')
    df_role = df_BLUE_RED[['champion_id_x', 'role_x', 'role_y', 'champion_id_y', 'win_x', 'game_id']]
    champion_vs_champion = pd.DataFrame(df_role[['champion_id_x', 'role_x', 'role_y', 'champion_id_y', 'win_x']].value_counts())

    #times that a given champion played against another champion by role
    total_champion_vs_champion = pd.DataFrame(df_BLUE_RED[['champion_id_x', 'role_x', 'role_y', 'champion_id_y']].value_counts())

    #percentage of times that a champion has lost or won against another champion
    rate_champion_vs_champion = champion_vs_champion.div(total_champion_vs_champion)

    rate_champion_vs_champion.to_csv('role_winrate_champ_vs_champ')

    return rate_champion_vs_champion
