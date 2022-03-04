import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

FILE_PATH = "../raw_data/full_dump.json"

def get_patch_year(s):
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
    #remove the games were a champion played in both teams -> not possible
    df_teams = df_teams[(df_teams.id != '2020 Mid-Season Cup/Scoreboards/Knockout Stage_1_1') & (df_teams.id != '2020 Mid-Season Cup/Scoreboards/Knockout Stage_3_1')]
    df_teams.rename(columns={"id": "game_id"})
    return df_teams[df_teams.year > 2015]


def get_data_split(split_value = 0.6):
    """Method to get the data splited in training and test"""

    df = get_data()

    df['patch'] = pd.to_numeric(df["patch"], downcast="float")
    df_sort = df.sort_values(['patch'], ascending=True)

    #Return data to train, test, score and evaluate
    ##Split_value*100% old games + halve new games to train
    data_length = int(len(df_sort)*split_value)
    data_sub_train = df_sort[:data_length]
    y = df_sort[data_length:-5]['winner']

    #get halve new games to train and test
    X_train, data_test, y_train, y_test = train_test_split(df_sort[data_length:-5],
                                                        y, test_size=0.5, random_state=42)

    #Get the 60% + 20% of data as training
    data_train = pd.concat([X_train, data_sub_train])

    ##Get the last 5 games for evaluating in the end
    data_eval = df_sort[-5:]

    #Return the train, test and eval data
    return data_train, data_eval, data_test


def process_json_data(df):
    #DataFrame for the teams stats
    df_teams = df.drop(['teams.BLUE.players', 'teams.RED.players', 'picks_bans'], axis=1)

    #DataFrame for the individual players stats plus the bans
    ##DataFrame dedicated to the BLUE team
    df['teams.BLUE.players'].explode()
    df_BLUE = pd.json_normalize(df['teams.BLUE.players'].explode())

    #DataFrame dedicated to the RED team
    df['teams.RED.players'].explode()
    df_RED = pd.json_normalize(df['teams.RED.players'].explode())

    #Include the game_id in every dataframe so we can merge dfs
    get_index = df['id'].tolist()
    index_preproc = np.asarray([[index] * 5 for index in get_index])
    index_teams = index_preproc.reshape(len(df) * 5).tolist()
    df_RED['game_id'] = index_teams
    df_BLUE['game_id'] = index_teams
    return df_teams, df_BLUE, df_RED


def get_train_data_only(BANS = False, evaluate_data = False, train_data = True, test_data = False):
    """Method to get the training/evaluate data divided into three DataFrames.
    Ban data is returned as a single DataFrame."""

    df_train, df_eval, df_test = get_data_split()

    if train_data:
        # roughly 80% of the data set to train the ML method
        df_teams, df_BLUE, df_RED = process_json_data(df_train)
        return df_teams, df_BLUE, df_RED

    if evaluate_data:
        # 5 data points to evaluate the method
        df_teams, df_BLUE, df_RED = process_json_data(df_eval)
        return df_teams, df_BLUE, df_RED

    if test_data:
        # rougly 20% of the data set to test the ML model
        df_teams, df_BLUE, df_RED = process_json_data(df_test)
        return df_teams, df_BLUE, df_RED

    if BANS:
        #Dataframe dedicated to the Bans
        df_train['picks_bans'].explode()
        df_BANS = pd.json_normalize(df_train['picks_bans'].explode())
        return df_BANS


def get_synergy(x,y, df_synergy_matrix):
    try:
        x = df_synergy_matrix.loc[x][y]
    except KeyError:
        x = 0.5
    return x


def pair_wise_synergy(df, df_synergy_matrix, name):
    #get the mean value of the pairwise combination of champions synergy
    result_Synergy = df[['game_id', 'champion_id']].groupby(['game_id']).aggregate({
        'champion_id': lambda z: [get_synergy(x,y, df_synergy_matrix) for x in z for y in z if x != y]
    })
    result_Synergy['mean_synergy_'+str(name)] = result_Synergy.champion_id.apply(lambda x: sum(x)/len(x))
    return result_Synergy.drop('champion_id', axis=1)


if __name__ == '__main__':
    df = get_data()
