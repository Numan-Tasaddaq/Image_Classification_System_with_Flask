import os
from flask import Blueprint, jsonify, render_template, request, redirect, url_for

api = Blueprint("api", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "image-classifier-flask"})

# Frontend: Home page
@api.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # (Later: pass to model for prediction)
        prediction = "digit: 7 (dummy prediction)"

        return render_template("index.html", prediction=prediction, filename=file.filename)

    return render_template("index.html", prediction=None, filename=None)
