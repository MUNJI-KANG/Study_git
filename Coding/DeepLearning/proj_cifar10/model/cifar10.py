import numpy as np
from tensorflow import keras
from PIL import Image

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 학습 모델 로딩.
model_path = r'./model/best-cifar10-deepcnn.keras'
model = keras.models.load_model(model_path)

def predict_images(img_paths):
    imgs = []

    for path in img_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array / 255.0
        imgs.append(img_array)

    batch_imgs = np.array(imgs)
    preds = model.predict(batch_imgs)
    preds_classes = [class_names[np.argmax(p)] for p in preds]

    return preds_classes