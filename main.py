from flask import Flask, request, jsonify
import os
import uuid
import cv2
from deepface import DeepFace
from mtcnn import MTCNN

app = Flask(__name__)
UPLOAD_FOLDER = "stored_faces"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = MTCNN()

# ---------- Utility: detect & crop face ----------
def detect_and_crop(image_path, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    faces = detector.detect_faces(img)
    if not faces:
        return None  # No face detected

    x, y, w, h = faces[0]["box"]
    face_crop = img[y:y+h, x:x+w]

    if save_path:
        cv2.imwrite(save_path, face_crop)

    return face_crop

# 1️⃣ Enroll Face
@app.route("/enroll", methods=["POST"])
def enroll_face():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(temp_path)

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Detect + Crop Face
    cropped = detect_and_crop(temp_path, filepath)
    os.remove(temp_path)

    if cropped is None:
        return jsonify({"error": "No face detected"}), 400

    return jsonify({"message": "Face enrolled", "filename": filename})

# 2️⃣ Match Face
@app.route("/match", methods=["POST"])
def match_face():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(temp_path)

    cropped = detect_and_crop(temp_path, temp_path)
    if cropped is None:
        os.remove(temp_path)
        return jsonify({"error": "No face detected"}), 400

    stored_images = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
    if not stored_images:
        os.remove(temp_path)
        return jsonify({"error": "No enrolled faces found"}), 404

    best_match = None
    best_score = 10  # smaller = better
    for img in stored_images:
        try:
            result = DeepFace.verify(temp_path, img, model_name="ArcFace", enforce_detection=False)
            if result["distance"] < best_score:
                best_score = result["distance"]
                best_match = os.path.basename(img)
        except Exception:
            continue

    os.remove(temp_path)

    if best_match:
        return jsonify({"match": True, "filename": best_match, "score": float(best_score)})
    else:
        return jsonify({"match": False}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
