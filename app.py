import json
import random
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import os

nltk.download('punkt')

stemmer = LancasterStemmer()

with open("intro.json") as file:
    data = json.load(file)

try:
    with open("models/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_y.append(intent["tag"])
            docs_x.append(wrds)

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []

    empty_out = [0 for x in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = empty_out[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    with open("models/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(training[0]),), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_path = "models/chat_model.keras"

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model: {e}. Training new model...")
        model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
        model.save(model_path)
else:
    print("Model not found. Training new model...")
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    model.save(model_path)

def bag_of_words(s, words):
    bag = [0 for x in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat(inp):
    input_data = np.array([bag_of_words(inp, words)])
    result = model.predict(input_data)[0]
    results_index = np.argmax(result)
    tag = labels[results_index]
    
    if result[results_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return random.choice(responses)
    else:
        return ("Sorry, I didn't understand that. Please change the question")

if __name__ == "__main__":
    print("Give input")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        response = chat(inp)
        print(response)