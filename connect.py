from flask import Flask, render_template, request, jsonify
from app import chat

connect = Flask(__name__)


@connect.get("/")
def index_get():
    return render_template("geminiVersion.html")


@connect.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = chat(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    connect.run(debug=True)
