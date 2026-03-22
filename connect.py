import os
from flask import Flask, render_template, request, jsonify
from app import chat

connect = Flask(__name__)


@connect.get("/")
def index_get():
    return render_template("index.html")


@connect.post("/predict")
def predict():
    data = request.get_json()
    if not data or not data.get("message"):
        return jsonify({"error": "No message provided"}), 400
    text = data["message"]
    response = chat(text)
    return jsonify({"answer": response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    connect.run(host="0.0.0.0", port=port)