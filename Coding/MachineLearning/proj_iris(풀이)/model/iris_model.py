import joblib, os
import pandas as pd

# 모델 & 스케일러 로드
model_path = os.path.join(r'./model/iris_model.pkl')
clf = joblib.load(model_path)

scaler_path = os.path.join(r'./model/iris_scaler.pkl')
scaler = joblib.load(scaler_path)

columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

# 예측 함수 작성
# 입력값: 웹에서 사용자가 입력한 값
# 출력값: 분류 문자열 (ex: Iris-setosa, Iris-versicolor, Iris-virginica)
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_scaled = scaler.transform(pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=columns))
    return clf.predict(input_scaled)[0]



    