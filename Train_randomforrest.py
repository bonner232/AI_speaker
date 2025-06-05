from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import librosa
import numpy as np
import os
import joblib
import matplotlib as plt
# Save the model
import noisereduce as nr
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    #mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #y_denoised = nr.reduce_noise(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)#
    # Extract MFCC features
    #mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13)

    return np.mean(mfcc.T, axis=0)  # Average over timen

# Load data
X = []
y = []

for file in os.listdir("TrainingData"):

    if file.endswith(".wav"):
        label = 0 if "andi" in file else 1  # adjust logic as needed

        features = extract_features(os.path.join("TrainingData", file))
        X.append(features)
        y.append(label)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print(y_pred)
print(classification_report(y_test, y_pred))


joblib.dump(clf, "voice_classifier_speech_final_no_noise_reduce2.0.joblib")
print("Model saved to voice_classifier_speech.joblib")

plt.figure(figsize=(40, 20))
plot_tree(clf.estimators_[0], feature_names=[f"mfcc_{j}" for j in range(len(X[0]))], filled=True)
plt.title("Tree 0")
plt.show()
