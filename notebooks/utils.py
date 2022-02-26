import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder


FILE_PATH = "../raw_data/full_dump.json"


def get_patch_year(s):
    """Transform the version of the game into the year"""
    return int(s.split('.')[0]) + 2010


def get_data():
    """Method to get the data"""

    #Open the file
    with open(FILE_PATH) as data_file:
        data = json.load(data_file)

    #Normalize the Json file
    df_normalized_teams = pd.json_normalize(data.values())
    df_teams = df_normalized_teams.copy()

    #return the year the game was played
    df_teams['year'] = df_teams.patch.apply(get_patch_year)
    df_teams = df_teams[(df_teams.id != '2020 Mid-Season Cup/Scoreboards/Knockout Stage_1_1') & (df_teams.id != '2020 Mid-Season Cup/Scoreboards/Knockout Stage_3_1')]
    return df_teams


def get_data_split(split_value = 0.8):
    """Method to get the data splited in training and test"""

    df = get_data()

    df['patch'] = pd.to_numeric(df["patch"], downcast="float")
    df_sort = df.sort_values(['patch'], ascending=True)

    #Return data to train, test, score and evaluate
    ##Data to train
    data_length = int(len(df_sort)/split_value)
    data_train = df_sort[:data_length]

    ##Get the last 5 games for evaluating in the end
    data_eval = df_sort[-5:]

    ##Data to test and score
    data_test = df_sort[data_length:-5]

    #Return the train, test and eval data
    return data_train, data_eval, data_test


def get_train_data_only(BANS = False, evaluate_data = False):
    """Method to get the ONLY training data divided into three DataFrames"""

    #DataFrame for the teams stats
    df_train, df_eval, df_test = get_data_split()
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

    if evaluate_data:
        return df_teams, df_BLUE, df_RED, df_BANS, df_eval

    return df_teams, df_BLUE, df_RED


if __name__ == '__main__':
    df = get_data()
