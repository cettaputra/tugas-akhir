from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import re
from PIL import Image, ImageOps
import io
import os
from gtts import gTTS
import random
import traceback
from datetime import datetime
import glob

app = Flask(__name__)
app.secret_key = "rahasia-cetta"

model = load_model('model/DeepSketch16-4.h5')

labels = [
    "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan",
    "lingkaran", "belah ketupat", "bintang", "segi delapan", "segi empat", "segitiga"
]

def preprocess_image(image):
    img = image.convert("L")
    img = ImageOps.invert(img)
    img_array = np.array(img)

    coords = np.argwhere(img_array > 30)
    if coords.size == 0:
        raise ValueError("Gambar terlalu kosong")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_array[y0:y1, x0:x1]

    size = max(cropped.shape)
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - cropped.shape[0]) // 2
    x_offset = (size - cropped.shape[1]) // 2
    square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped

    processed = Image.fromarray(square).resize((28, 28), Image.Resampling.LANCZOS)
    processed_array = np.array(processed) / 255.0
    processed_array = processed_array.reshape(1, 28, 28, 1)

    return processed_array

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/play')
def play():
    if 'round' not in session:
        session['round'] = 1
        session['score'] = 0
        session['labels'] = random.sample(labels, 5)
        session['drawn_images'] = []

    if session['round'] > 5:
        return redirect(url_for('result'))

    label = session['labels'][session['round'] - 1]
    session['current_target'] = label

    tts = gTTS(f"Yuk, kita gambar {label}", lang='id')
    audio_path = os.path.join("static", "tts.mp3")
    tts.save(audio_path)

    timestamp = datetime.now().timestamp()
    audio_url = f"/static/tts.mp3?v={timestamp}"

    return render_template('play.html', label=label, audio_url=audio_url)

@app.route('/predict_live', methods=['POST'])
def predict_live():
    try:
        if session.get('round', 6) > 5:
            return jsonify({'redirect': '/result'})

        data = request.get_json()

        # Jika timeout dikirim dari frontend
        if data.get("timeout", False):
            session['round'] += 1
            if session['round'] > 5:
                return jsonify({'redirect': '/result'})
            else:
                return jsonify({'redirect': '/play'})

        img_str = re.search(r'base64,(.*)', data['image']).group(1)
        decoded = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(decoded))

        processed = preprocess_image(img)
        pred = model.predict(processed)
        prediction = labels[np.argmax(pred)]

        correct = prediction == session.get('current_target', '')

        if correct:
            session['score'] += 1

            hasil_folder = os.path.join("static", "img", "hasil")
            os.makedirs(hasil_folder, exist_ok=True)
            existing_files = sorted(glob.glob(os.path.join(hasil_folder, "*.png")), key=os.path.getctime)
            if len(existing_files) >= 10:
                os.remove(existing_files[0])

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = f"hasil_{timestamp}.png"
            path = os.path.join(hasil_folder, filename)
            img.save(path)

            session['drawn_images'].append({
                "image": f"/static/img/hasil/{filename}",
                "label": prediction
            })

            session['round'] += 1

            if session['round'] > 5:
                return jsonify({'correct': True, 'prediction': prediction, 'redirect': '/result'})
            else:
                return jsonify({'correct': True, 'prediction': prediction, 'redirect': '/play'})

        # Salah, tetap di sesi yang sama, tidak tambah score
        return jsonify({'correct': False, 'prediction': prediction})

    except Exception as e:
        print("[ERROR]", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    results = session.get('drawn_images', [])
    score = session.get('score', 0)

    tts = gTTS(f"Skor kamu adalah {score} dari lima", lang='id')
    audio_path = os.path.join("static", "tts_hasil.mp3")
    tts.save(audio_path)

    timestamp = datetime.now().timestamp()
    audio_url = f"/static/tts_hasil.mp3?v={timestamp}"

    return render_template('result.html', results=results, score=score, audio_url=audio_url)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

@app.route('/speech', methods=['GET', 'POST'])
def speech():
    current = session.get('speech_index', 0)
    drawn_images = session.get('drawn_images', [])

    if current >= len(drawn_images):
        return redirect(url_for('index'))  # selesai semua

    item = drawn_images[current]
    label = item['label']
    image = item['image']

    # Generate audio TTS
    tts = gTTS(f"Gambar ini adalah {label}. Yuk, kita ucapkan: {label}", lang='id')
    audio_path = os.path.join("static", "tts_speech.mp3")
    tts.save(audio_path)
    timestamp = datetime.now().timestamp()
    audio_url = f"/static/tts_speech.mp3?v={timestamp}"

    return render_template('speech.html', image=image, label=label, audio_url=audio_url)

@app.route('/next_speech', methods=['POST'])
def next_speech():
    session['speech_index'] = session.get('speech_index', 0) + 1
    return redirect(url_for('speech'))

