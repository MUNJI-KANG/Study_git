# uvicorn main:app --reload

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from model.fashionmnist import *
import os, shutil

app = FastAPI()


#템플릿 설정
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates", auto_reload=True)


# 'uploads' 폴더를 '/static' 경로로 접근할 수 있도록 설정
# 예: uploads/image.jpg 파일은 http://127.0.0.1:8000/static/image.jpg 로 접근 가능
UPLOAD_DIR = 'uploads'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount('/static', StaticFiles(directory=UPLOAD_DIR))

@app.get('/')
def home(request : Request):
    return templates.TemplateResponse('index.html', {"request" : request})

@app.post('/upload')
def upload_image(fashionfile : UploadFile = File(...)):
    # print('🐎', fashionfile.filename)
    file_path = os.path.join(UPLOAD_DIR, fashionfile.filename) #저장 파일 경로
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(fashionfile.file,f)


    # 예측 동작
    pred = predict(file_path)
    print('🦄', pred)

    # 업로드 성공 후, 접근 가능한 이미지 URL을 함께 반환
    image_url = f'/static/{fashionfile.filename}'


    return {
        "filename" : fashionfile.filename,
        'url' : image_url,
        'pred' : pred,
        }