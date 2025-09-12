import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# User input
filepath = tf.keras.utils.get_file(input("Please enter the URL to the source text: "))
text = open(filepath, 'rb').read().decode(encoding='utf-8', errors='ignore').lower()

m = int(input("Please enter from where splice should begin: "))
n = int(input("Please enter where splice should end: "))
model_name = input("Please enter the name of the model: ")

# Slice text
text = text[m:n]

# Ensure enough length
SEQ_LENGTH = 40
STEP_SIZE = 3
if len(text) < SEQ_LENGTH:
    raise ValueError("Text too short for given SEQ_LENGTH.")

# Mapping characters
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

# Prepare sequences
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Model
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001))

model.fit(x, y, batch_size=128, epochs=5)

import json

# Save mappings
with open(f"{model_name}_mappings.json", "w") as f:
    json.dump({
        "char_to_index": char_to_index,
        "index_to_char": {str(i): c for i, c in index_to_char.items()}
    }, f)


# Save model
model.save(f"{model_name}.keras")
