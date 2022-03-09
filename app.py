import json

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('LOL Winner Prediction')

roles = ["TOP", "JGL", "MID", "BOT", "SUP"]

# Riot APIs for LoL
# https://developer.riotgames.com/docs/lol
api_version = "12.4.1"
champions_url = f"https://ddragon.leagueoflegends.com/cdn/{api_version}/data/en_US/champion.json"
image_url = f"http://ddragon.leagueoflegends.com/cdn/{api_version}/img/champion/"

@st.cache
def load_data(filename):
    data = pd.read_json(filename)
    return data

champions_data = load_data(champions_url)


c1, c2, c3, c4 = st.columns((2, 1, 1, 2))

# Default initial values
blue_team = {role: champion for (role, champion) in zip(roles, champions_data.index)}
red_team = {role: champion for (role, champion) in zip(roles, champions_data.index[5:])}

# c1.subheader('Blue Team')

for role in roles:
    champions = [champion for champion in champions_data.index if champion not in list(blue_team.values()) + list(red_team.values())]
    blue_team[role] = c1.selectbox(
        "Blue " + role,
        champions
    )

for blue_champion in blue_team.values():
    c2.image(image_url + blue_champion + ".png", width=80)


# st.write('You selected team:', blue_team["TOP"])

# c4.subheader('Red Team')

for role in roles:
    champions = [champion for champion in champions_data.index if champion not in list(blue_team.values()) +  list(red_team.values())]
    red_team[role] = c4.selectbox(
        "Red " + role,
        champions
    )

for red_champion in red_team.values():
    c3.image(image_url + red_champion + ".png", width=80)


# st.write('You selected team:', blue_team["TOP"])

def predict(top_blue,   #Blue team top line champion
            jgl_blue,   #Blue team jungler champion
            bot_blue,   #Blue team bottom line champion
            mid_blue,   #Blue team middle line champion
            sup_blue,   #Blue team support champion
            top_red,    #Red team top line champion
            jgl_red,    #Red team jungler champion
            bot_red,    #Red team bottom line champion
            mid_red,    #Red team middle line champion
            sup_red):   #Red team support champion

    #Predcitions are done in terms of the champion_id not the name
    ##Function get_champion_id searchs and returns the id of champions
    X = pd.DataFrame({
        'TOP_x' : top_blue,
        'JGL_x' : jgl_blue,
        'BOT_x' : bot_blue,
        'MID_x' : mid_blue,
        'SUP_x' : sup_blue,
        'TOP_y' : top_red,
        'JGL_y' : jgl_red,
        'BOT_y' : bot_red,
        'MID_y' : mid_red,
        'SUP_y' : sup_red,
        }, index=[0])

    #pipeline previously trained with the test data
    pipeline = joblib.load('model.joblib')

    #make prediction
    results = pipeline.predict(X)
    prediction = float(results[0])

    return dict(winner=prediction)

#st.button
if st.button('Predict Winner'):
    #st.write(predict('Annie', 'Annie', 'Annie', 'Annie', 'Annie', 'Annie', 'Annie', 'Annie', 'Annie', 'Annie'))
    st.write(predict(blue_team['TOP'], blue_team['JGL'], blue_team['MID'], blue_team['BOT'], blue_team['SUP'],
        red_team['TOP'], red_team['JGL'], red_team['MID'], red_team['BOT'], red_team['SUP']))
else:
    st.write('Goodbye')
