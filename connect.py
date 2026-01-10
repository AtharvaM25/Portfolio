from flask import Flask, render_template, request, jsonify
from app import chat

connect = Flask(__name__)


@connect.get("/")
def index_get():
    return render_template("index.html")


@connect.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = chat(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    connect.run(host="0.0.0.0", port=port)

