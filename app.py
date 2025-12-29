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

print(f"DEBUG: Base Directory is: {BASE_DIR}")
print(f"DEBUG: Looking for models in: {MODELS_DIR}")

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    print(f"DEBUG: Creating 'models' directory at {MODELS_DIR}")
    os.makedirs(MODELS_DIR)

if os.path.exists(MODELS_DIR):
    print(f"DEBUG: 'models' folder contents: {os.listdir(MODELS_DIR) if os.listdir(MODELS_DIR) else 'EMPTY'}")

# Download NLTK data
try:
    nltk.download('punkt', download_dir=NLTK_DIR, quiet=True)
    nltk.data.path.append(NLTK_DIR)
except Exception as e:
    print(f"NLTK download warning: {e}")

stemmer = LancasterStemmer()

json_path = os.path.join(BASE_DIR, "intro.json")
with open(json_path) as file:
    data = json.load(file)

def train_model():
    print("=" * 50)
    print("🚀 TRAINING NEW MODEL - This will take a few minutes...")
    print("=" * 50)
    
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
    
    # Ensure directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Save data
    with open(DATA_FILE, "wb") as f:
        pickle.dump((words, labels, training, output), f)
    print("✅ Training data saved")
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(len(training[0]),), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(len(output[0]), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("🔄 Training model (this may take 2-3 minutes)...")
    # Reduced epochs for faster training on deployment
    model.fit(training, output, epochs=500, batch_size=8, verbose=0)
    
    # Save model with error handling
    try:
        model.save(MODEL_FILE)
        print(f"✅ Model saved successfully to: {MODEL_FILE}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise
    
    print("=" * 50)
    
    return words, labels, training, output, model

# Initialize variables
words = None
labels = None
training = None
output = None
model = None

# Check if both files exist before trying to load
data_exists = os.path.exists(DATA_FILE)
model_exists = os.path.exists(MODEL_FILE)

print(f"📊 Data file exists: {data_exists}")
print(f"🤖 Model file exists: {model_exists}")

if data_exists and model_exists:
    # Both files exist, try to load them
    try:
        print("📂 Attempting to load existing model and data...")
        
        with open(DATA_FILE, "rb") as f:
            words, labels, training, output = pickle.load(f)
        print("✅ Data loaded successfully")
        
        model = tf.keras.models.load_model(MODEL_FILE)
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"⚠️  Error loading existing files: {e}")
        print("🔄 Will train new model...")
        words, labels, training, output, model = train_model()
else:
    # Files don't exist, train new model
    print("⚠️  Model or data files not found")
    print("🔄 Starting training process...")
    words, labels, training, output, model = train_model()

# Verify everything loaded correctly
if words is None or labels is None or model is None:
    raise Exception("Failed to initialize model and data!")

print("\n✅ Chatbot initialization complete!")


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
    """Main chat function called by Flask app"""
    try:
        input_data = np.array([bag_of_words(inp, words)])
        result = model.predict(input_data, verbose=0)[0]
        results_index = np.argmax(result)
        tag = labels[results_index]
        
        if result[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            return random.choice(responses)
        else:
            return "Sorry, I didn't understand that. Please rephrase your question."
    except Exception as e:
        print(f"Error in chat function: {e}")
        return "Sorry, I'm having trouble processing that right now."


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("💬 CHATBOT READY - You can start chatting!")
    print("=" * 50 + "\n")
    print("Type 'quit' to exit\n")
    
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        response = chat(inp)
        print(f"Bot: {response}")