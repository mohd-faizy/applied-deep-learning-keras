# ==============================
# 1. Imports & Setup
# ==============================
import warnings
warnings.filterwarnings("ignore")
import os

import tensorflow as tf
from tensorflow.keras.layers import (
    TextVectorization, Embedding, GRU, Dense, Dropout, Input
)
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

# Download NLTK resources quietly
for resource in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ==============================
# 2. Load Data
# ==============================
# index_col=0 handles the leading comma (ID column) in your header
data = pd.read_csv("_datasets/clothing_review.csv", index_col=0)

# Filter for rows where Class Name exists
data = data[data["Class Name"].notna()]

# ==============================
# 3. Label & Text Processing
# ==============================
# Create Binary Target: 1 if Rating > 3, else 0
y = data["Rating"].apply(lambda x: 1 if x > 3 else 0)

def clean_text(text):
    if isinstance(text, float) or pd.isna(text): 
        return ""
    # Remove special chars and lowercase
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    # Lemmatize and remove stopwords
    return " ".join(lemm.lemmatize(w) for w in text.split() if w not in stop_words)

# Combine Title + Review + Class Name
data["Cleaned_Text"] = (
    data["Title"].fillna("") + " " + 
    data["Review Text"].fillna("") + " " + 
    data["Class Name"]
).apply(clean_text)

# ==============================
# 4. Train-Test Split (The Keras 3 Fix)
# ==============================
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    data["Cleaned_Text"], y, test_size=0.25, random_state=42
)

# --- CRITICAL FIX START ---
# Keras 3 cannot handle NumPy string arrays directly. 
# We must convert them to TensorFlow Tensors and reshape them to (Batch_Size, 1).

# 1. Convert to TF Tensor
X_train_tensor = tf.convert_to_tensor(X_train_raw.tolist(), dtype=tf.string)
X_test_tensor = tf.convert_to_tensor(X_test_raw.tolist(), dtype=tf.string)

# 2. Reshape to (N, 1) to match Input(shape=(1,))
X_train = tf.reshape(X_train_tensor, [-1, 1])
X_test = tf.reshape(X_test_tensor, [-1, 1])

# 3. Convert targets to float32 Tensors
y_train = tf.convert_to_tensor(y_train_raw.values, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test_raw.values, dtype=tf.float32)
# --- CRITICAL FIX END ---

# ==============================
# 5. Vectorization
# ==============================
VOCAB_SIZE = 10000
MAX_LEN = 50

vectorizer = TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN)
# Adapt using the tensor (use the flattened version for adapt)
vectorizer.adapt(X_train_tensor)

# ==============================
# 6. Model Definition
# ==============================
model = Sequential([
    # Explicitly expect a single string per input
    Input(shape=(1,), dtype=tf.string),
    
    vectorizer,
    
    Embedding(input_dim=VOCAB_SIZE, output_dim=128),
    GRU(64, return_sequences=True),
    GRU(64),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ==============================
# 7. Training
# ==============================
print("Starting training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=5, 
    validation_split=0.2, 
    batch_size=32
)

# ==============================
# 8. Evaluation
# ==============================
eval_loss, eval_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {eval_acc:.4f}")

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()


# ==============================
# 9. Prediction on New Data
# ==============================
def predict_sentiment(reviews):
    """
    Takes a list of raw strings, cleans them, formats them 
    for Keras 3 inputs, and returns predictions.
    """
    print("\n--- Processing New Predictions ---")
    
    # 1. Apply the SAME text cleaning used during training
    # Note: If you want to include "Class Name" like in training, 
    # append it to the string before passing it here.
    cleaned_reviews = [clean_text(r) for r in reviews]
    
    # 2. Convert to Tensor (Keras 3 requirement)
    # We must convert the list to a Tensor, then reshape to (Batch_Size, 1)
    review_tensor = tf.convert_to_tensor(cleaned_reviews, dtype=tf.string)
    review_inputs = tf.reshape(review_tensor, [-1, 1])
    
    # 3. Predict
    # Result is a probability (0 to 1)
    predictions = model.predict(review_inputs)
    
    # 4. Display Results
    for review, score in zip(reviews, predictions):
        # Threshold is 0.5 because we used Sigmoid
        label = "Positive (>3 stars)" if score > 0.5 else "Negative (<=3 stars)"
        confidence = score[0] if score > 0.5 else 1 - score[0]
        
        print(f"\nReview: {review}")
        print(f"Prediction: {label}")
        print(f"Score: {score[0]:.4f}")
        print("-" * 30)

# --- Test with Custom Examples ---
sample_reviews = [
    "I absolutely loved this dress, the fabric is wonderful and fits perfectly!",
    "Terrible quality. The zipper broke immediately and the color is faded.",
    "It is okay, not the best but good for the price. Sizing is a bit off.",
    "The material feels very cheap and scratchy. Returning it."
]

predict_sentiment(sample_reviews)