from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import librosa
import numpy as np
import os
import joblib
import matplotlib as plt
# Save the model


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return np.mean(mfcc.T, axis=0)  # Average over time

# Load data
X = []
y = []

for file in os.listdir("training"):
    if file.endswith(".wav"):
        label = 0 if "andi" in file else 1  # adjust logic as needed
        features = extract_features(os.path.join("training", file))
        X.append(features)
        y.append(label)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print(y_pred)
print(classification_report(y_test, y_pred))


joblib.dump(clf, "voice_classifier.joblib")
print("Model saved to voice_classifier.joblib")

