import numpy as np
from tensorflow import keras
import PIL.ImageOps as ops 
from PIL import Image

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

# 학습모델 로딩
model_path = r'./model/best-cnn-model.keras'
model = keras.models.load_model(model_path)

# 입력 : 파일경로
# 출력 : 분류 클래스
def predict(file_path):
  img = Image.open(file_path)
  mono8img = img.convert('L')
  invImg = ops.invert(mono8img)
  resizeImg = invImg.resize((28, 28))
  data_arr = np.array(resizeImg).reshape(1, 28, 28, 1)
  data_scaled = data_arr / 255.
  preds = model.predict(data_scaled)
  return classes[np.argmax(preds)]



