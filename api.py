from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def index(**params):
    params = ['TOP_x', 'JGL_x','BOT_x','MID_x','SUP_x', 'TOP_y', 'JGL_y','BOT_y','MID_y','SUP_y']
    return model.predict(params)
