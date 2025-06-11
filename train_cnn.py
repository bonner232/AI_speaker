import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models


def extract_mfcc(file_path, n_mfcc=13, max_len=100):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to fixed length (max_len frames)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = mfcc.T  # (time_steps, n_mfcc)
    mfcc = mfcc[..., np.newaxis]  # (time_steps, n_mfcc, 1)

    return mfcc


# Prepare data
X = []
y = []

data_dir = 'TrainingData'  # your folder with wav files
for filename in os.listdir(data_dir):
    if filename.endswith('.wav'):
        label = 0 if 'andi' in filename else 1
        mfcc = extract_mfcc(os.path.join(data_dir, filename))
        X.append(mfcc)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)  # Expect (num_samples, time_steps, n_mfcc, 1)
print("Labels shape:", y.shape)

# Define model
input_shape = X[0].shape  # (time_steps, n_mfcc, 1)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model (adjust epochs/batch size as needed)
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)
model.save('voice_classifier_cnn.keras')
print("Model saved to voice_classifier_cnn.keras")

import matplotlib.pyplot as plt

history = model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Training vs. Validation Accuracy')
plt.show()