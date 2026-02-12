# uvicorn iris_main:app --reload
from fastapi import FastAPI, Request, Body
from model.iris_model import *

# 템플릿 설정
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates", auto_reload=True)


app = FastAPI()  # FastAPI 인스턴스 생성

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("irisform.html", {"request": request, "slen_min":1.0, "slen_max":2.0})

from pydantic import BaseModel 
class InputIris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/iris")
def class_iris(body: InputIris):
    print('✅ body: ', body)
    result = predict_iris(body.sepal_length, body.sepal_width, body.petal_length, body.petal_width)
    return {"result": result}