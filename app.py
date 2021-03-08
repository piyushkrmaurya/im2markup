import json
from io import BytesIO

import cv2
import numpy as np
import requests
from flask import Flask, render_template, request

from preprocess import preprocess_image
from translate import generate_latex

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sample")
def sample():
    return render_template("sample.html")


@app.route("/api/", methods=["POST"])
def api():
    if request.get_json() and "url" in request.get_json():
        url = request.get_json().get("url")

        image_bytes = BytesIO()
        response = requests.get(url, stream=True)

        if response.ok:
            for block in response.iter_content(1024):
                if not block:
                    break
                image_bytes.write(block)
        img = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), -1)

    elif request.files.get("file", None):
        f = request.files["file"]
        img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    else:
        return "Error"

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = preprocess_image(img)
    img.save("client.png")
    latex = generate_latex(img)
    return latex


if __name__ == "__main__":
    app.static_url_path="/data"
    app.static_folder="data"
    app.run()
