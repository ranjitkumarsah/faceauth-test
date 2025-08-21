import os
import uuid
from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Folder to store enrolled faces
ENROLL_DIR = "faces"
os.makedirs(ENROLL_DIR, exist_ok=True)


@app.route("/")
def home():
    return jsonify({"message": "Face Recognition API is running!"})


# 1. Enroll API
@app.route("/enroll", methods=["POST"])
def enroll_face():
    try:
        if "image" not in request.files or "user_id" not in request.form:
            return jsonify({"error": "Send 'image' file and 'user_id' field"}), 400

        image = request.files["image"]
        user_id = request.form["user_id"]

        # Save image with unique name under user's folder
        user_folder = os.path.join(ENROLL_DIR, user_id)
        os.makedirs(user_folder, exist_ok=True)

        file_path = os.path.join(user_folder, f"{uuid.uuid4().hex}.jpg")
        image.save(file_path)

        return jsonify({"message": f"Face enrolled for {user_id}", "path": file_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 2. Match API
@app.route("/match", methods=["POST"])
def match_face():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Send 'image' file"}), 400

        query_img = request.files["image"]
        query_path = os.path.join("/tmp", f"{uuid.uuid4().hex}.jpg")
        query_img.save(query_path)

        # Compare with all enrolled images
        results = []
        for root, dirs, files in os.walk(ENROLL_DIR):
            for file in files:
                enrolled_path = os.path.join(root, file)
                try:
                    result = DeepFace.verify(query_path, enrolled_path, model_name="ArcFace")
                    if result["verified"]:
                        user_id = os.path.basename(root)
                        results.append({
                            "user_id": user_id,
                            "distance": result["distance"],
                            "threshold": result["threshold"]
                        })
                except Exception:
                    continue

        os.remove(query_path)

        if not results:
            return jsonify({"match": False, "message": "No matching face found"})

        # Sort by best (lowest distance)
        best = sorted(results, key=lambda x: x["distance"])[0]

        return jsonify({"match": True, "best_match": best, "candidates": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
