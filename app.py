from flask import Flask, render_template, request, redirect, url_for
import torch
from PIL import Image
import io
import base64
import cv2
import numpy as np

app = Flask(__name__)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'imageData' in request.form:
            # 이미지 데이터 수신
            image_data = request.form['imageData'].split(',')[1]
            img_bytes = io.BytesIO(base64.b64decode(image_data))
            img = Image.open(img_bytes).convert('RGB')
            img_np = np.array(img)

            # 모델 예측
            results = model(img_np)
            boxes = results.xyxy[0].numpy()

            # 바운딩 박스 그리기
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_np, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 바운딩 박스를 포함한 이미지를 base64로 인코딩
            _, img_encoded = cv2.imencode('.png', img_np)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            return render_template('result.html', image_data=img_base64)
        else:
            return redirect(url_for('index'))
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True, ssl_context=('cert.pem', 'key.pem'))
