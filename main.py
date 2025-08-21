from flask import Flask, request, jsonify
import os
from deepface import DeepFace
import cv2
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "stored_faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1️⃣ Enroll Face
@app.route("/enroll", methods=["POST"])
def enroll_face():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return jsonify({"message": "Face enrolled", "filename": filename})

# 2️⃣ Match Face
@app.route("/match", methods=["POST"])
def match_face():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    temp_path = "temp.jpg"
    file.save(temp_path)

    stored_images = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
    if not stored_images:
        return jsonify({"error": "No enrolled faces found"}), 404

    best_match = None
    best_score = 10  # smaller = better
    for img in stored_images:
        try:
            result = DeepFace.verify(temp_path, img, model_name="ArcFace", enforce_detection=False)
            if result["distance"] < best_score:
                best_score = result["distance"]
                best_match = os.path.basename(img)
        except Exception as e:
            continue

    if best_match:
        return jsonify({"match": True, "filename": best_match, "score": best_score})
    else:
        return jsonify({"match": False}), 200
