import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import pickle

# Load intents
with open('data/intents.json') as file:
    data = json.load(file)

# Prepare training data
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        words.extend(pattern.split())
        docs_x.append(pattern)
        docs_y.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = sorted(list(set(words)))
labels = sorted(labels)

# Create training data
training = []
output = []
out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = [1 if word in doc.split() else 0 for word in words]
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training, output, epochs=200, batch_size=5, verbose=1)

# Save model and data
model.save('model/chatbot_model.h5')
with open('model/words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('model/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print("Training complete!")
