from flask import Flask, render_template, request, jsonify, redirect
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "image-classifier-flask"})

@app.route("/", methods=["GET", "POST"])
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

        # Dummy prediction (later replaced with real model)
        prediction = "digit: 7 (dummy prediction)"

        return render_template("index.html", prediction=prediction, filename=file.filename)

    return render_template("index.html", prediction=None, filename=None)

if __name__ == "__main__":
    app.run(debug=True)
