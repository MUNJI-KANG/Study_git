from model.iris_model import *
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI() 

# 템플릿 생성
templates = Jinja2Templates(directory="templates", auto_reload=True)


# static 파일 설정
app.mount("/static", StaticFiles(directory="static"), name='static')


#처음 로딩시 html 불러오기
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("irisform.html", {"request": request})

class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 사용자가 post 방식으로 요청했을경우
@app.post("/predict")
def class_iris(body : InputData):
    print('🌹 확인 : ', body)
    # 기존에 만들어둔 함수를 이용하여 predict 진행
    result = predict_iris(**body.model_dump())

    return {"result" :  result}