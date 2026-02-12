
# 모델 & 스케일러 로드
# 저장된 파일 확인
import joblib, os

base_dir = r"F:\KDT2508\Dropbox\K01\PyWork\MachineLearning\proj_iris\model"

model_path = os.path.join(base_dir, "iris_model.pkl")
scaler_path = os.path.join(base_dir, "iris_scaler.pkl")

loaded_clf = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# 예측 함수 작성
# 입력값: 웹에서 사용자가 입력한 값
# 출력값: 분류 문자열 (ex: Iris-setosa, Iris-versicolor, Iris-virginica)
def predict_iris(sepal_length, sepal_width, petal_length, petal_width) -> str:
    x = [[sepal_length, sepal_width, petal_length, petal_width ]]
    x_scaled = loaded_scaler.transform(x)
    pred = loaded_clf.predict(x_scaled)[0]

    return str(pred)