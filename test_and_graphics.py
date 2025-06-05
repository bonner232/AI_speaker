import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
# === Load Trained Model ===
clf = joblib.load("voice_classifier_speech_final_no_noise_reduce2.0_20mfcc.joblib")

# === Extract Features from New WAV File ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    #y_denoised = nr.reduce_noise(y=y, sr=sr)
    #mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)#13
    return np.mean(mfcc.T, axis=0)

# === Predict ===
def predict_voice(file_path):
    features = extract_features(file_path).reshape(1, -1)
    label = clf.predict(features)[0]
    return "andi" if label == 0 else "miro"

# Example usage
file_to_predict = "output_1.wav"
result = predict_voice(file_to_predict)
print(f"Prediction: {result}")



importances = clf.feature_importances_
feature_names = [f"MFCC{i+1}" for i in range(len(importances))]




plt.figure(figsize=(10, 4))
plt.bar(feature_names, importances)
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()