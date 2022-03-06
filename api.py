import pandas as pd
import requests

from fastapi import FastAPI


app = FastAPI()
url = 'https://ddragon.leagueoflegends.com/cdn/12.3.1/data/en_US/champion.json'
resp = requests.get(url=url)
data = resp.json()


def get_champion_id(name, data):
    champion_info = pd.DataFrame(data['data'])
    return champion_info.loc['key', name]


@app.get("/")
def index():
    return dict(greeting="hello")


@app.get("/predict")
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
    X = pd.DataFrame(dict(
        'TOP_x' = get_champion_id(top_blue, data),
        'JGL_x' = get_champion_id(jgl_blue, data),
        'BOT_x' = get_champion_id(bot_blue, data),
        'MID_x' = get_champion_id(mid_blue, data),
        'SUP_x' = get_champion_id(sup_blue, data),
        'TOP_y' = get_champion_id(top_red, data),
        'JGL_y' = get_champion_id(jgl_red, data),
        'BOT_y' = get_champion_id(bot_red, data),
        'MID_y' = get_champion_id(mid_red, data),
        'SUP_y' = get_champion_id(sup_red, data)
        )

    # pipeline previously trained with the test data
    pipeline = joblib.load('model.joblib')

    #make prediction
    results = pipeline.predict(X)
    prediction = float(results[0])

    return dict(winner=prediction)
