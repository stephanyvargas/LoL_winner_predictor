import json

import streamlit as st
import pandas as pd
import numpy as np

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
