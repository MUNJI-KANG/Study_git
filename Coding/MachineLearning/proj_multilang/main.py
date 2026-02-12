from fastapi import FastAPI, Request
import joblib, os

#모델 로딩
model_path = os.path.join(r'./model/freq.pkl')
clf = joblib.load(model_path)

def detect_lang(text):
    # 알파벳 출현 빈도 구하기
    text = text.lower()


    # 알파벳인 경우 해당 알파벳의 빈도수 ++
    cnt = []
    for i in range(26):
        ch = chr(i + ord('a'))
        cnt.append(text.count(ch))
       
    total = sum(cnt)
   
    if total == 0: return "입력이 없습니다"
   
    freq = list(map(lambda n: n / total, cnt))
   
    # 언어 예측하기
    res = clf.predict([freq])
   
    # 언어 코드를 한국어로 변환하기
    lang_dic = {
        "en": "영어",
        "fr": "프랑스어",
        "id": "인도네시아어",
        "tl": "타갈로그어"
    }
    return lang_dic[res[0]]




# 템플릿
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates", auto_reload = True)




app = FastAPI()

# static 파일 설정
from fastapi.staticfiles import StaticFiles
app.mount("/static",StaticFiles(directory="static"), name='static')

@app.get("/") # GET 요청 "/"을 받으면 처리하는 함수 
def root():
    return {"message"  : "하랑이 월드"}  # 기본적으로 JSON 형태로 응답.



# 실행
# uvicorn main:app --reload
# uvicorn main:app --reload --port 8001


@app.get("/lang")
def form(request:Request):
    # return{"name":"초롱하랑", "age":25}
    return templates.TemplateResponse('langform.html', {'request' : request, 'title' : '외국어 문장 분류'})
from pydantic import BaseModel

class InputText(BaseModel):
    text:str # name='text'

@app.post('/detect_lang')
def class_lang(body: InputText):
    print('🐱body : ', body)
    result = detect_lang(body.text)
        
    return {"result": result}