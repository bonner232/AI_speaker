import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import wave
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib
from numba.np.arrayobj import array_ndim
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import Gui_livedetection
from Gui_livedetection import switch

loaded_wav_files = []

# Recording variables
is_recording = False
recorded_frames = []
samplerate = 44100  # Standard sample rate
switch = True


stop_event = threading.Event()

def start_live_classification():
    set_led_color()

    # Start the worker in a new thread
    stop_event.clear()  # reset stop signal
    thread = threading.Thread(target=live_loop)
    thread.daemon = True  # Optional: thread exits with main program
    thread.start()

def stop_live_classification():
    stop_event.set()  # Signal the loop to stop

def live_loop():
    print("Live classification started.")
    while not stop_event.is_set():
        print("Classifying...")
        ergraw = Gui_livedetection.live_val()
        erg = list(ergraw)
        print("-------------")
        print(erg)

    print("Live classification stopped.")
'''
def start_live_classification():
    while switch:

        erg=Gui_livedetection.live_val()
        print("-------------")
        print(erg)

def stop_live_classification():
    Gui_livedetection.switch = False
    return False
    
    
    '''
def show_info():
    messagebox.showinfo("How it works",
                        "This tool lets you load multiple WAV files.\nYou can select, view, plot, and process them.\nResults of processing will appear below.\n\n"
                        "You can also record audio using the Start and Stop buttons.")

def is_valid_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            wf.getparams()
        return True
    except wave.Error:
        return False

def select_wav_file():
    file_paths = filedialog.askopenfilenames(
        title="Select WAV file(s)",
        filetypes=[("WAV files", "*.wav")]
    )
    if not file_paths:
        messagebox.showwarning("No file", "No file was selected!")
        return

    new_files = 0
    for file_path in file_paths:
        if file_path not in loaded_wav_files:
            if is_valid_wav(file_path):
                loaded_wav_files.append(file_path)
                listbox.insert(tk.END, os.path.basename(file_path))
                new_files += 1
            else:
                messagebox.showerror("Invalid File", f"'{os.path.basename(file_path)}' is not a valid WAV file.")
        else:
            messagebox.showinfo("Duplicate", f"'{os.path.basename(file_path)}' is already loaded.")

    if new_files:
        messagebox.showinfo("Files Loaded", f"{new_files} new valid WAV file(s) loaded.")

def get_selected_files():
    selected_indices = listbox.curselection()
    return [loaded_wav_files[i] for i in selected_indices]

def plot_selected_files():
    selected_files = get_selected_files()
    if not selected_files:
        messagebox.showwarning("No Selection", "Please select one or more files from the list.")
        return

    for file_path in selected_files:
        try:
            with wave.open(file_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                signal = wf.readframes(n_frames)
                waveform = np.frombuffer(signal, dtype=np.int16)

                if n_channels == 2:
                    waveform = waveform[::2]

                times = np.linspace(0, n_frames / framerate, num=n_frames)

                plt.figure(figsize=(10, 4))
                plt.plot(times, waveform)
                plt.title(os.path.basename(file_path))
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not plot '{os.path.basename(file_path)}':\n{e}")

def process_wav_file(file_path):
    try:
        clf = joblib.load("voice_classifier_speech_final_no_noise_reduce2.0.joblib")#wihtout nois ereduction it works better
        y, sr = librosa.load(file_path, sr=None)
        #y_denoised = nr.reduce_noise(y=y, sr=sr)  # try with and without noise reduction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        #mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13)
        mfcc_cleaned = np.mean(mfcc.T, axis=0).reshape(1, -1)

        label = clf.predict(mfcc_cleaned)[0]
        probabilities = clf.predict_proba(mfcc_cleaned)[0]
        confidence = max(probabilities) * 100  # as a percentage

        result = "andi" if label == 0 else "miro"
        print(f"Processing: {file_path} => {result} ({confidence:.2f}%)")

        result_box.insert(tk.END, f"{os.path.basename(file_path)}: {result} ({confidence:.2f}%)\n")
        result_box.see(tk.END)

    except Exception as e:
        print(f"Error processing {file_path}:", e)
        result_box.insert(tk.END, f"Error: {os.path.basename(file_path)} - {str(e)}\n")
        result_box.see(tk.END)

def process_selected_files():
    result_box.delete("1.0", tk.END)
    selected_files = get_selected_files()
    if not selected_files:
        messagebox.showwarning("No Selection", "Please select one or more files from the list.")
        return

    for file_path in selected_files:
        process_wav_file(file_path)

# --- Audio Recording Functions ---
def record_audio():
    global is_recording, recorded_frames
    is_recording = True
    recorded_frames = []

    def callback(indata, frames, time, status):
        if is_recording:
            recorded_frames.append(indata.copy())
        else:
            raise sd.CallbackAbort

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        while is_recording:
            sd.sleep(100)

def start_recording():
    threading.Thread(target=record_audio).start()
    messagebox.showinfo("Recording", "Recording started...")

def stop_recording():
    global is_recording
    is_recording = False
    if recorded_frames:
        audio = np.concatenate(recorded_frames, axis=0)
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if save_path:
            write(save_path, samplerate, audio)
            messagebox.showinfo("Saved", f"Audio saved to:\n{save_path}")
            if is_valid_wav(save_path):
                loaded_wav_files.append(save_path)
                listbox.insert(tk.END, os.path.basename(save_path))

# --- GUI Setup ---
root = tk.Tk()
root.title("WAV File Selector")
root.geometry("450x600")

# Menu
menubar = Menu(root)
help_menu = Menu(menubar, tearoff=0)
help_menu.add_command(label="How it works", command=show_info)
menubar.add_cascade(label="Help", menu=help_menu)
root.config(menu=menubar)

# Buttons
tk.Button(root, text="Add WAV File(s)", command=select_wav_file, width=20, height=2).pack(pady=10)

# Listbox
listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50, height=10)
listbox.pack(pady=5)

# Action buttons
tk.Button(root, text="Show Selected File(s)",
          command=lambda: messagebox.showinfo("Selected", "\n".join(get_selected_files()))).pack(pady=5)
tk.Button(root, text="Plot Selected WAV File(s)", command=plot_selected_files).pack(pady=5)
tk.Button(root, text="Predict WAV File", command=process_selected_files).pack(pady=5)

# Recording buttons
tk.Button(root, text="Start Recording", command=start_recording, width=20).pack(pady=5)
tk.Button(root, text="Stop & Save Recording", command=stop_recording, width=20).pack(pady=5)

tk.Button(root, text="Start Live Classification", command=start_live_classification, width=25).pack(pady=5)
tk.Button(root, text="Stop Live", command=stop_live_classification, width=25).pack(pady=5)

# Result box
tk.Label(root, text="Processing Results:").pack()
result_box = tk.Text(root, width=50, height=8, bg="lightyellow")
result_box.pack(pady=5)

#led

# Label as a colored box (LED)
led = tk.Label(root, width=10, height=5, bg="gray")  # size and default color
led.pack(pady=20)

# Function to change LED color
def set_led_color():
    led.config(bg="green")



root.mainloop()
