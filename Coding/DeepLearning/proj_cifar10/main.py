# uvicorn main:app --reload

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles  # 1. 추가
from typing import List
from model.cifar10 import *
import os
import shutil

app = FastAPI()

# static 파일 설정
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 설정
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates", auto_reload=True)

# 'uploads' 폴더를 '/upload' 경로로 서빙하도록 설정
# 예: uploads/image.jpg 파일은 http://127.0.0.1:8000/files/image.jpg 로 접근 가능
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ↓ multiple 파일(들)을 imagefiles 라는 name으로 받아온다
@app.post("/upload")
async def upload_image(imagefiles: List[UploadFile] = File(...)):  

    files_name = []
    files_path = []

    for file in imagefiles:
        # 확인용
        print('🤖 업로드', file.filename)

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    
        files_name.append(file.filename)
        files_path.append(file_path)

    preds = predict_images(files_path)
    for pred in preds:
        print('📌', pred)

    image_urls = [f"/files/{name}" for name in files_name]

    return{
        'urls' : image_urls,
        'preds' : preds
    }
        
