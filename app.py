import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from model import predict_image  # Your PyTorch prediction function

app = Flask(__name__)

# ------------------ Upload folder ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ Home route ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            prediction = predict_image(filepath)
        except Exception as e:
            prediction = f"Error: {str(e)}"

        return render_template("index.html", filename=filename, prediction=f"Digit: {prediction}")

    return render_template("index.html", filename=None, prediction=None)

# ------------------ REST API route ------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return {"error": "Invalid file"}, 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        prediction = predict_image(filepath)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True)
