import sounddevice as sd
from scipy.io.wavfile import write
import os
import sys
import joblib
import librosa
import numpy as np
from time import sleep
from datetime import datetime

probarr1=[]
labelarr1=[]
switch = True

# Resource path handling
def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

model_path = get_resource_path("voice_classifier_speech_final_no_noise_reduce2.0.joblib")

def extraction_live(filename, duration=1, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio)


def live_classifier(filename):
    clf = joblib.load(model_path)

    # Extract features
    y, sr = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_cleaned = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # Predict label
    label = clf.predict(mfcc_cleaned)[0]

    # Predict probability if possible
    try:
        proba = clf.predict_proba(mfcc_cleaned)[0]
        confidence = np.max(proba)
        class_names = clf.classes_ if hasattr(clf, "classes_") else ["0", "1"]
        predicted_class = class_names[label]
        print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
        print(f"Class probabilities: {dict(zip(class_names, proba))}")

        prob1 = proba[0]
        prob2 = proba[1]
        if prob1>prob2:


            labelarr1.append(label)
            probarr1.append(prob1)
        else:
            labelarr1.append(label)
            probarr1.append(prob2)

    except AttributeError:
        print(f"Prediction: {'andi' if label == 0 else 'miro'} (No confidence available)")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Raw label: {label}")
    print("")

    return int(label)



def cal_avg():
    outputarr=[]
    labelarr = [int(x) for x in labelarr1]
    probarr = [float(x) for x in probarr1]




    print(labelarr)
    print(probarr)
    zeros = 0
    ones = 0
    for j in labelarr:
        if j==1:
            ones = ones + 1
        else:
            zeros = zeros + 1
    if ones>zeros:
        print("miro")
        for k in labelarr:
            if k == 0:
                labelarr.pop(k)
                probarr.pop(k)

    else:
        print("andi")
        for k in labelarr:
            if k == 1:
                labelarr.pop(k)
                probarr.pop(k)

    all_val = 0
    print(f"probarr:{probarr}")
    for l in probarr:

        all_val = l + all_val
    average = all_val / len(probarr)
    print(f"avg{average} and label {labelarr[0]}")
    return average, labelarr[0]






def live_val():



    for i in range(3):
        print(f"\n--- Iteration {i + 1}/30 ---")
        filename = f"output_{i + 1}.wav"
        extraction_live(filename)
        live_classifier(filename)



    return cal_avg()




