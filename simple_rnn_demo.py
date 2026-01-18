import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ==========================================
# 1. Load and Visualize the Data
# ==========================================
# We use the Monthly Milk Production dataset. 
# It's a simple sequence: Time vs Production.
# Perfect for learning how RNNs capture temporal patterns.

data_path = "_datasets/monthly-milk-production-pounds.csv"
print(f"Loading data from {data_path}...")

df = pd.read_csv(data_path, index_col='Month', parse_dates=True)

# Rename column for easier access
df.columns = ['Production']

# Let's verify the data loaded
print(df.head())

# Plot the data to see the trend and seasonality
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Production'])
plt.title("Monthly Milk Production")
plt.xlabel("Date")
plt.ylabel("Production (Pounds)")
plt.show()

# ==========================================
# 2. Data Preprocessing
# ==========================================
# RNNs work best with normalized data (usually between 0 and 1 or -1 and 1).
# We also need to split into Train and Test sets *chronologically*
# because this is time-series data (we can't shuffle!).

test_size = 12  # We will attempt to predict the last 12 months
train_data = df.iloc[:-test_size]
test_data = df.iloc[-test_size:]

scaler = MinMaxScaler()
scaler.fit(train_data) # Fit only on training data to avoid data leakage

scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)

# ==========================================
# 3. Format Data for RNN (The "Sliding Window")
# ==========================================
# An RNN looks at a sequence of 'n_input' steps to predict the next step.
# For example, if n_input=3:
# Input: [Month1, Month2, Month3] -> Target: Month4
# Input: [Month2, Month3, Month4] -> Target: Month5

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

n_input = 12  # Use past 12 months to predict the next 1 month
n_features = 1 # We only have 1 feature (Production)

# Create the generator
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

print(f"Generator created. Input sequence length: {n_input}")
# Let's peek at the first batch to understand it
X, y = generator[0]
print("First Input Sequence (normalized):", X.flatten())
print("First Target (normalized):", y)

# ==========================================
# 4. Build the RNN Model
# ==========================================
# We use a 'SimpleRNN' layer. 
# It processes the sequence step-by-step, maintaining a 'hidden state' (memory).

model = Sequential()

# SimpleRNN layer
# units=100: Number of neurons in the hidden state
# activation='relu': Activation function
# input_shape=(n_input, n_features): (12 time steps, 1 feature per step)
model.add(SimpleRNN(100, activation='relu', input_shape=(n_input, n_features)))

# Dense output layer
# We just want 1 number prediction (the next month's production)
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print("\nModel Summary:")
model.summary()

# ==========================================
# 5. Train the Model
# ==========================================
print("\nTraining the model...")
model.fit(generator, epochs=50, verbose=1)

# ==========================================
# 6. Evaluate / Predict
# ==========================================
# Now we use the trained model to predict the test sequence (the last 12 months).
# Note: To predict the *first* test point, we need the *last 12 points* of the training data.

test_predictions = []

# Start with the last 'n_input' points from the training set
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):
    # Get the prediction value for the first batch
    current_pred = model.predict(current_batch, verbose=0)[0]
    
    # Append the prediction into the array
    test_predictions.append(current_pred) 
    
    # Update the batch to include the new prediction and drop the oldest point
    # current_batch[:, 1:, :] is all points except the first one
    # current_pred is the new point to add
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform predictions back to original scale (pounds)
true_predictions = scaler.inverse_transform(test_predictions)

# ==========================================
# 7. Compare and Visualize
# ==========================================
test_data['Predictions'] = true_predictions

print("\nPrediction Results:")
print(test_data)

plt.figure(figsize=(12, 8))
plt.plot(train_data.index, train_data['Production'], label='Train Data')
plt.plot(test_data.index, test_data['Production'], label='Real Test Data')
plt.plot(test_data.index, test_data['Predictions'], label='RNN Predictions', linestyle='--')
plt.legend()
plt.title("RNN Performance: Real vs Predicted")
plt.show()
