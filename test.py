from fastapi import FastAPI, status, Response

app = FastAPI()

@app.get('/')
def home():
    return {"Value": "This is a simple API"}

def calculate(x,y):
    # model prediction
    return x+y

@app.get('/hackathon/{x}/{y}')
def home(x: int, y:int):
    pred = calculate(x,y)
    return {"Value": pred}