from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import random
import cv2

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('my_model.h5')

class_names = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    detected_objects = []
    threshold = 0.2

    for idx, prob in enumerate(predictions):
        if prob > threshold:
            box = {
                "x": random.randint(50, 100),
                "y": random.randint(50, 100),
                "width": random.randint(350, 350),
                "height": random.randint(300, 300)
            }
            detected_objects.append({
                "class_name": class_names[idx],
                "confidence": float(prob),
                "box": box
            })

    return jsonify({'objects': detected_objects})


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    method = request.form.get('method', 'thresholding')
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img_cv = np.array(img)

    if img_cv.shape[2] == 4:
        img_cv = img_cv[:, :, :3]

    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    if method == 'thresholding':
        _, processed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    elif method == 'edge':
        processed = cv2.Canny(img_gray, 100, 200)

    elif method == 'region':
        processed = cv2.GaussianBlur(img_gray, (11, 11), 0)

    elif method == 'features':
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
        processed = cv2.magnitude(sobelx, sobely)
        processed = np.uint8(np.clip(processed, 0, 255))

    elif method == 'clustering':
        Z = img_cv.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        processed = centers[labels.flatten()].reshape(img_cv.shape)

    else:
        return jsonify({'error': 'Unknown method'}), 400

    pil_image = Image.fromarray(processed)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_io = io.BytesIO()
    pil_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
