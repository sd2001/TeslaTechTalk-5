from fastapi import FastAPI, status, Response
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn, json, requests
from pydantic import BaseModel
import uuid
import bcrypt
from main import *
from bson import ObjectId
from bson.json_util import loads, dumps

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI(debug = True,
                title="TeslaTechTalks",
                description="API endpoints for CIFAR10 Predictions",
                version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)             

class ImageLink(BaseModel):
    url: str = None

class ImagePred(BaseModel):
    id_: str = None
    prediction: str = None

def parse_json(data):
    return json.loads(dumps(data))

@app.get("/Image/Categories", status_code = 200, name = "Get Category list")
async def showjobs(response:Response):
    return categories

@app.post("/Image/Predict", status_code = 200, name = "Get Predictions")
async def showjobs(response:Response, link: ImageLink):
    try:
        url = link.url
        y = predict(url)
        id_ = uuid.uuid1()
        ip = ImagePred()
        ip.id_ = id_
        ip.prediction = y
        response.status_code = status.HTTP_200_OK
        return parse_json(ip.__dict__)
    except Exception as e:
        return status.HTTP_404_NOT_FOUND

@app.get("/")
async def home(request: Request):
    return {"Value": "Server Up and Running"}


# if __name__ == '__main__':
# uvicorn.run(app, port=8080)