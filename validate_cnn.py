import numpy as np
import librosa
import tensorflow as tf

def extract_mfcc(file_path, n_mfcc=13, max_len=100):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    mfcc = mfcc.T
    mfcc = mfcc[..., np.newaxis]

    return mfcc

# Load your saved model
model = tf.keras.models.load_model('voice_classifier_cnn.keras')
print("Model loaded.")

# Load and preprocess new audio file
new_file = 'UnseenData/andi.wav'  # Replace with your file path
mfcc = extract_mfcc(new_file)
mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension: (1, time_steps, n_mfcc, 1)
print(mfcc.shape)

# Predict
prediction = model.predict(mfcc)
print(f"Prediction score: {prediction[0][0]:.4f}")

# Interpret prediction
if prediction[0][0] > 0.5:
    print("Predicted: Your miro")
else:
    print("Predicted: andi")
