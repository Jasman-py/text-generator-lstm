import numpy as np
import random
import tensorflow as tf
import json
from tensorflow.keras.models import load_model

# Load model
model_name =input("Enter previously saved model name: ")
model = load_model(f"{model_name}.keras")

# Load saved mappings
with open(f"{model_name}_mappings.json", "r") as f:
    mappings = json.load(f)

char_to_index = {c: int(i) for c, i in mappings["char_to_index"].items()}
index_to_char = {int(i): c for i, c in mappings["index_to_char"].items()}

SEQ_LENGTH = 40  # must match training

def sample(preds, temperature=1.0):
    """Sample an index from probability array with temperature scaling."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, length=400, temperature=0.5):
    """Generate text from trained model given a seed string."""
    generated = seed_text

    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(char_to_index)), dtype=bool)
        for t, char in enumerate(seed_text[-SEQ_LENGTH:]):
            if char in char_to_index:
                x[0, t, char_to_index[char]] = 1.0

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        seed_text = seed_text[1:] + next_char  # shift window

    return generated

# Example usage
seed = input("Enter a seed text: ").lower()
print("\nGenerated Text:\n")
print(generate_text(seed, length=500, temperature=0.7))
