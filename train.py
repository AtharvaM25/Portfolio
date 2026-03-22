import json
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'chat_model.h5')
DATA_FILE = os.path.join(MODELS_DIR, 'data.pickle')
NLTK_DIR = os.path.join(BASE_DIR, 'nltk_data')

nltk.data.path.insert(0, NLTK_DIR)

stemmer = LancasterStemmer()

with open(os.path.join(BASE_DIR, "intro.json")) as f:
    data = json.load(f)

words, labels, docs_x, docs_y = [], [], [], []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
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
print("data.pickle saved")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(training[0]),), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training, output, epochs=500, batch_size=8, verbose=1)
model.save(MODEL_FILE,save_format='h5')
print(f"chat_model.keras saved")