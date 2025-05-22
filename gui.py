import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import wave
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib
from numba.np.arrayobj import array_ndim

loaded_wav_files = []


def show_info():
    messagebox.showinfo("How it works",
                        "This tool lets you load multiple WAV files.\nYou can select, view, plot, and process them.\nResults of processing will appear below.")


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
        # Load model
        clf = joblib.load("voice_classifier.joblib")

        # Extract features
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_cleand = np.mean(mfcc.T, axis=0).reshape(1, -1)  

        # Predict
        label = clf.predict(mfcc_cleand)[0]
        result = "andi" if label == 0 else "miro"

        # Output
        print(f"Processing: {file_path} => {result}")
        result_box.insert(tk.END, f"{os.path.basename(file_path)}: {result}\n")
        result_box.see(tk.END)

    except Exception as e:
        print(f"Error processing {file_path}:", e)
        result_box.insert(tk.END, f"Error: {os.path.basename(file_path)} - {str(e)}\n")
        result_box.see(tk.END)

def process_selected_files():
    result_box.delete("1.0", tk.END)  # Clear previous results
    selected_files = get_selected_files()
    if not selected_files:
        messagebox.showwarning("No Selection", "Please select one or more files from the list.")
        return

    for file_path in selected_files:
        process_wav_file(file_path)


# GUI setup
root = tk.Tk()
root.title("WAV File Selector")
root.geometry("450x550")

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
tk.Button(root, text="predict wav file", command=process_selected_files).pack(pady=5)

# Result box
tk.Label(root, text="Processing Results:").pack()
result_box = tk.Text(root, width=50, height=8, bg="lightyellow")
result_box.pack(pady=5)

root.mainloop()
