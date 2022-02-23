import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FILE_PATH = "../raw_data/full_dump.json"


def get_data():
    """Method to get the data"""

    #Open the file
    with open(FILE_PATH) as data_file:
        data = json.load(data_file)

    #Normalize the Json file
    df_normalized_teams = pd.json_normalize(data.values())
    df_teams = df_normalized_teams.copy()

    return df_teams


def get_data_split():
    """Method to get the data splited in training and test"""

    df = get_data()
    y = LabelEncoder().fit(df.winner).transform(df.winner)
    X = df.drop('winner', axis=1)

    #Return the train and test data
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_train_data_only(BANS = False):
    """Method to get the ONLY training data divided into three DataFrames"""

    #DataFrame for the teams stats
    df_train, _, _, _ = get_data_split()
    df_teams = df_train.drop(['teams.BLUE.players', 'teams.RED.players', 'picks_bans'], axis=1)

    #DataFrame for the individual players stats plus the bans
    ##DataFrame dedicated to the BLUE team
    df_train['teams.BLUE.players'].explode()
    df_BLUE = pd.json_normalize(df_train['teams.BLUE.players'].explode())

    #DataFrame dedicated to the RED team
    df_train['teams.RED.players'].explode()
    df_RED = pd.json_normalize(df_train['teams.RED.players'].explode())

    #Include the game_id in every dataframe so we can merge dfs
    get_index = df_train['id'].tolist()
    index_preproc = np.asarray([[index] * 5 for index in get_index])
    index_teams = index_preproc.reshape(len(df_train) * 5).tolist()
    df_RED['game_id'] = index_teams
    df_BLUE['game_id'] = index_teams

    if BANS:
        #Dataframe dedicated to the Bans
        df_train['picks_bans'].explode()
        df_BANS = pd.json_normalize(df_train['picks_bans'].explode())
        return df_teams, df_BLUE, df_RED, df_BANS

    return df_teams, df_BLUE, df_RED


if __name__ == '__main__':
    df = get_data()
