from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import json
import random

app = Flask(__name__)

# Load model and data
model = load_model('model/chatbot_model.h5')
with open('model/words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('model/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
with open('data/intents.json') as file:
    intents = json.load(file)

# Chatbot prediction function
def predict_intent(text):
    bag = [1 if word in text.split() else 0 for word in words]
    prediction = model.predict(np.array([bag]))[0]
    max_index = np.argmax(prediction)
    tag = labels[max_index]
    return tag if prediction[max_index] > 0.7 else "unknown"

def get_response(tag):
    if tag == "unknown":
        return "Sorry, I didn’t understand that. Can you rephrase?"
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_chatbot_response():
    user_text = request.form['msg']
    tag = predict_intent(user_text.lower())
    response = get_response(tag)
    return response

if __name__ == '__main__':
    app.run(debug=True)
