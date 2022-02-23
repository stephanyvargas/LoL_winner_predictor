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
    df_teams = df_train.copy()

    #DataFrame for the individual players stats plus the bans
    ##DataFrame dedicated to the BLUE team
    df_teams['teams.BLUE.players'].explode()
    df_normalized_BLUE = pd.json_normalize(df_teams['teams.BLUE.players'].explode())
    df_BLUE = df_normalized_BLUE.copy()

    #DataFrame dedicated to the RED team
    df_teams['teams.RED.players'].explode()
    df_normalized_RED = pd.json_normalize(df_teams['teams.RED.players'].explode())
    df_RED = df_normalized_RED.copy()

    if BANS:
        #Dataframe dedicated to the Bans
        df_teams['picks_bans'].explode()
        df_normalized_BANS = pd.json_normalize(df_teams['picks_bans'].explode())
        df_BANS = df_normalized_BANS.copy()
        df_teams = df_train.drop(['teams.BLUE.players', 'teams.RED.players', 'picks_bans'], axis=1)
        return df_teams, df_BLUE, df_RED, df_BANS

    df_teams = df_train.drop(['teams.BLUE.players', 'teams.RED.players', 'picks_bans'], axis=1)
    return df_teams, df_BLUE, df_RED


if __name__ == '__main__':
    df = get_data()
