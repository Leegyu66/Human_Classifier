import io
from flask import Flask, jsonify, request, render_template
from PIL import Image
from matplotlib.pyplot import imshow
import torch
import torch.nn as nn

from torchvision import transforms, models

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = model = torch.load("model.pt", map_location=device)

class_names = ['김종국', '마동석', '이병헌']
# 이미지를 읽어 결과를 반환하는 함수
def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        print(_)
        # imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

    return class_names[preds[0]]

app = Flask(__name__)

@app.route('/')
def template(name = None):
    return render_template('template.html', name=name)

@app.route('/fileupload', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # 이미지 바이트 데이터 받아오기
        file = request.files['file']
        image_bytes = file.read()

        # 분류 결과 확인 및 클라이언트에게 결과 반환
        class_name = get_prediction(image_bytes=image_bytes)
        return render_template('file.html', value=class_name)

if __name__=='__main__':
    app.run(debug=True)