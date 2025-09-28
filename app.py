import os
import sqlite3
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import cv2, base64
import numpy as np
from datetime import datetime
import easyocr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger YOLO et EasyOCR
model = YOLO("best.pt")
reader = easyocr.Reader(['en'])  # tu peux ajouter 'fr' si besoin

# ---------- DB ----------
DB_NAME = "history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    action TEXT
                )''')
    conn.commit()
    conn.close()

def save_history(plate, confidence, action):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (plate, confidence, timestamp, action) VALUES (?, ?, ?, ?)",
              (plate, confidence, timestamp, action))
    conn.commit()
    conn.close()

# ---------- Detection ----------
def detect_license_plate(image):
    img_np = np.array(image)
    results = model(img_np)[0]
    detections = []

    if results.boxes is not None:
        for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.conf.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)

            # --- Crop de la plaque ---
            plate_crop = img_np[y1:y2, x1:x2]

            # --- OCR sur la plaque ---
            plate_text = ""
            if plate_crop.size > 0:  # éviter crash si crop vide
                ocr_result = reader.readtext(plate_crop)
                if len(ocr_result) > 0:
                    plate_text = ocr_result[0][1]  # texte détecté

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': round(float(conf), 3),
                'class': int(cls),
                'plate_text': plate_text
            })

    annotated_img = results.plot()
    return detections, annotated_img

@app.route('/')
def dashboard():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = Image.open(filepath).convert('RGB')
    detections, annotated_img = detect_license_plate(image)

    # Sauvegarde dans historique
    for d in detections:
        save_history(d['plate_text'], d['confidence'], "entry")

    # Conversion image annotée en base64
    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', annotated_img_bgr)
    img_str = base64.b64encode(buffer).decode('utf-8')

    accuracy = round(sum(d['confidence'] for d in detections) / len(detections), 3) if detections else 0.0

    return jsonify({
        'detections': detections,
        'accuracy': accuracy,
        'annotated_image': f'data:image/jpeg;base64,{img_str}'
    })

@app.route('/history')
def history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT plate, confidence, timestamp, action FROM history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
