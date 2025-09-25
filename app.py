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
import torch

# -------------------- CONFIG FLASK --------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------- PATCH TORCH >= 2.6 --------------------
try:
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules.conv as conv
    import ultralytics.nn.modules.head as head
    import ultralytics.nn.modules.block as block
    import torch.nn.modules.container as container

    if hasattr(torch, "serialization"):
        torch.serialization.add_safe_globals([
            YOLO,
            tasks.DetectionModel,     # architecture YOLO
            container.Sequential,     # nn.Sequential
            conv.Conv,                # couches convolution
            head.Detect,              # tête de détection
            block.C2f                 # blocs YOLOv8
        ])
except Exception as e:
    print("⚠️ Patch Torch échoué :", e)

# -------------------- CHARGEMENT YOLO --------------------
model = None
try:
    if os.path.exists("best.pt"):
        model = YOLO("best.pt")
        print("✅ Modèle YOLO chargé")
    else:
        print("⚠️ Fichier best.pt introuvable ! Place-le dans ton repo GitHub.")
except Exception as e:
    print("❌ Erreur chargement modèle :", e)

# -------------------- CHARGEMENT OCR --------------------
try:
    reader = easyocr.Reader(['en'], gpu=False)  # ⚠️ Render n'a pas de GPU
except Exception as e:
    print("❌ Erreur EasyOCR :", e)
    reader = None

# -------------------- BASE DE DONNÉES --------------------
DB_NAME = "history.db"

def init_db():
    try:
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
    except Exception as e:
        print("❌ Erreur init DB :", e)

def save_history(plate, confidence, action):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO history (plate, confidence, timestamp, action) VALUES (?, ?, ?, ?)",
                  (plate, confidence, timestamp, action))
        conn.commit()
        conn.close()
    except Exception as e:
        print("❌ Erreur sauvegarde DB :", e)

# -------------------- DETECTION --------------------
def detect_license_plate(image):
    if model is None:
        return [], np.array(image)

    img_np = np.array(image)
    try:
        results = model(img_np)[0]
    except Exception as e:
        print("❌ Erreur YOLO inference :", e)
        return [], img_np

    detections = []
    if results.boxes is not None:
        for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.conf.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)

            # --- Crop plaque ---
            plate_crop = img_np[y1:y2, x1:x2]
            plate_text = ""

            if plate_crop.size > 0 and reader:
                try:
                    ocr_result = reader.readtext(plate_crop)
                    if len(ocr_result) > 0:
                        plate_text = ocr_result[0][1]
                except Exception as e:
                    print("⚠️ Erreur OCR :", e)

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': round(float(conf), 3),
                'class': int(cls),
                'plate_text': plate_text
            })

    try:
        annotated_img = results.plot()
    except Exception:
        annotated_img = img_np

    return detections, annotated_img

# -------------------- ROUTES --------------------
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

    # Sauvegarde historique
    for d in detections:
        save_history(d['plate_text'], d['confidence'], "entry")

    # Conversion en base64
    try:
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', annotated_img_bgr)
        img_str = base64.b64encode(buffer).decode('utf-8')
    except Exception:
        img_str = ""

    accuracy = round(sum(d['confidence'] for d in detections) / len(detections), 3) if detections else 0.0

    return jsonify({
        'detections': detections,
        'accuracy': accuracy,
        'annotated_image': f'data:image/jpeg;base64,{img_str}'
    })

@app.route('/history')
def history():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT plate, confidence, timestamp, action FROM history ORDER BY id DESC LIMIT 50")
        rows = c.fetchall()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": f"Erreur DB: {e}"}), 500

# -------------------- LANCEMENT --------------------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 10000))  # ⚠️ Render impose PORT
    app.run(host="0.0.0.0", port=port)
