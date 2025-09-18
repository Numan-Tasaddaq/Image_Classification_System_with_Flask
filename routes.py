import os
from flask import Blueprint, jsonify, render_template, request, redirect, url_for, current_app
from werkzeug.utils import secure_filename
from model import predict_image

# ------------------ Blueprint ------------------
api = Blueprint("api", __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ Health check ------------------
@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "image-classifier-flask"})

# ------------------ Frontend route ------------------
@api.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)

        # ------------------ Save file inside static/uploads dynamically ------------------
        upload_folder = os.path.join(current_app.root_path, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # ------------------ Predict digit ------------------
        try:
            prediction = predict_image(filepath)
        except Exception as e:
            prediction = f"Error: {str(e)}"

        return render_template(
            "index.html",
            prediction=f"Digit: {prediction}",
            filename=filename
        )

    return render_template("index.html", prediction=None, filename=None)

# ------------------ REST API route ------------------
@api.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    # ------------------ Save file inside static/uploads dynamically ------------------
    upload_folder = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # ------------------ Predict digit ------------------
    try:
        prediction = predict_image(filepath)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
