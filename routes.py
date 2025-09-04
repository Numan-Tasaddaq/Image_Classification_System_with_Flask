import os
from flask import Blueprint, jsonify, render_template, request, redirect, url_for
from model import predict_image

api = Blueprint("api", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "image-classifier-flask"})

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

        # Predict with model
        prediction = predict_image(filepath)

        return render_template("index.html", prediction=f"digit: {prediction}", filename=file.filename)

    return render_template("index.html", prediction=None, filename=None)
