import json
import random
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'chat_model.keras')
DATA_FILE = os.path.join(MODELS_DIR, 'data.pickle')
NLTK_DIR = os.path.join(BASE_DIR, 'nltk_data')

# Tell NLTK to use the committed nltk_data folder — no download needed
nltk.data.path.insert(0, NLTK_DIR)

stemmer = LancasterStemmer()

json_path = os.path.join(BASE_DIR, "intro.json")
with open(json_path) as file:
    data = json.load(file)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def load_model():
    """Load pre-trained model and data. Raises clearly if files are missing."""
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "Pre-trained model files not found.\n"
            "Run this locally first to generate them:\n"
            "  python app.py --train\n"
            "Then commit models/chat_model.keras and models/data.pickle to your repo."
        )

    print("Loading model and data...")
    with open(DATA_FILE, "rb") as f:
        words, labels, training, output = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_FILE)
    print("Model loaded successfully!")
    return words, labels, model


def train_and_save():
    """Train the model locally and save files. Run this before deploying."""
    print("Training model...")
    words, labels, docs_x, docs_y = [], [], [], []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_y.append(intent["tag"])
            docs_x.append(wrds)
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = sorted(set(stemmer.stem(w.lower()) for w in words if w != '?'))
    labels = sorted(labels)

    training, output = [], []
    empty_out = [0] * len(labels)

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)
        output_row = empty_out[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(DATA_FILE, "wb") as f:
        pickle.dump((words, labels, training, output), f)
    print("Training data saved.")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(len(training[0]),), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(len(output[0]), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(training, output, epochs=500, batch_size=8, verbose=1)
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print("You can now commit the models/ folder and deploy.")


# Load model at import time (used by connect.py / gunicorn)
words, labels, model = load_model()


def chat(inp):
    """Main chat function called by Flask app."""
    try:
        input_data = np.array([bag_of_words(inp, words)])
        result = model.predict(input_data, verbose=0)[0]
        results_index = np.argmax(result)
        tag = labels[results_index]

        if result[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    return random.choice(tg['responses'])

        return "Sorry, I didn't understand that. Please rephrase your question."
    except Exception as e:
        print(f"Error in chat function: {e}")
        return "Sorry, I'm having trouble processing that right now."


if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        train_and_save()
    else:
        print("Usage: python app.py --train")
        print("Run this locally to generate model files before deploying.")