import tkinter as tk
from tkinter import filedialog, messagebox, Menu, ttk
import wave
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import Gui_livedetection
from PIL import Image, ImageTk
import matplotlib

import tkinter as tk
from tkinter import filedialog, messagebox, Menu, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import wave
import os
import librosa
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import threading

matplotlib.use('TkAgg')


# Resource path handling
def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)


model_path = get_resource_path("voice_classifier_speech_final_no_noise_reduce2.0_20_pre_finalfcc.joblib")

# Global variables
loaded_wav_files = []
is_recording = False
recorded_frames = []
samplerate = 44100
stop_event = threading.Event()

# Modern color scheme
BG_COLOR = "#2d2d2d"
ACCENT_COLOR = "#3498db"
SECONDARY_COLOR = "#2980b9"
TEXT_COLOR = "#ecf0f1"
BUTTON_COLOR = "#3c3c3c"
ENTRY_COLOR = "#3c3c3c"
HIGHLIGHT_COLOR = "#1abc9c"
LISTBOX_COLOR = "#3c3c3c"
RESULT_BOX_COLOR = "#2c3e50"


# Style configuration
def configure_styles():
    style = ttk.Style()
    style.theme_use('clam')

    # Configure frame style
    style.configure('TFrame', background=BG_COLOR)

    # Configure label style
    style.configure('TLabel',
                    background=BG_COLOR,
                    foreground=TEXT_COLOR,
                    font=('Segoe UI', 10))

    # Configure button style
    style.configure('TButton',
                    background=BUTTON_COLOR,
                    foreground=TEXT_COLOR,
                    font=('Segoe UI', 9, 'bold'),
                    borderwidth=1,
                    relief='flat')

    style.map('TButton',
              background=[('active', SECONDARY_COLOR), ('pressed', ACCENT_COLOR)],
              foreground=[('active', TEXT_COLOR), ('pressed', TEXT_COLOR)])

    # Configure listbox style
    style.configure('Listbox',
                    background=LISTBOX_COLOR,
                    foreground=TEXT_COLOR,
                    selectbackground=ACCENT_COLOR,
                    font=('Segoe UI', 9),
                    borderwidth=0,
                    highlightthickness=0)

    # Configure scrollbar style
    style.configure('Vertical.TScrollbar',
                    background=BG_COLOR,
                    troughcolor=BG_COLOR,
                    bordercolor=BG_COLOR,
                    arrowcolor=TEXT_COLOR,
                    gripcount=0)


# LED indicator class
class LEDIndicator(tk.Canvas):
    def __init__(self, parent, size=20, **kwargs):
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, bd=0, **kwargs)
        self.size = size
        self.led_color = "gray"
        self.draw_led()

    def draw_led(self):
        self.delete("all")
        padding = 2
        self.create_oval(padding, padding, self.size - padding,
                         self.size - padding, fill=self.led_color,
                         outline="", tags="led")

    def set_color(self, color):
        self.led_color = color
        self.draw_led()


# Audio functions
def is_valid_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            wf.getparams()
        return True
    except wave.Error:
        return False


def select_wav_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    initial_directory = os.path.join(script_dir, "UnseenData")
    file_paths = filedialog.askopenfilenames(
        title="Select WAV file(s)",
        filetypes=[("WAV files", "*.wav")],
        initialdir=initial_directory
    )

    if not file_paths:
        return

    for file_path in file_paths:
        if file_path not in loaded_wav_files and is_valid_wav(file_path):
            loaded_wav_files.append(file_path)
            listbox.insert(tk.END, os.path.basename(file_path))
            status_var.set(f"Loaded: {os.path.basename(file_path)}")


def get_selected_files():
    selected_indices = listbox.curselection()
    return [loaded_wav_files[i] for i in selected_indices]


def plot_selected_files():
    selected_files = get_selected_files()
    if not selected_files:
        status_var.set("Please select files to plot")
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

                plt.figure(figsize=(10, 4), facecolor='#3c3c3c')
                ax = plt.axes()
                ax.set_facecolor('#3c3c3c')
                ax.tick_params(colors=TEXT_COLOR)
                ax.xaxis.label.set_color(TEXT_COLOR)
                ax.yaxis.label.set_color(TEXT_COLOR)
                ax.title.set_color(TEXT_COLOR)

                plt.plot(times, waveform, color=HIGHLIGHT_COLOR)
                plt.title(os.path.basename(file_path), color=TEXT_COLOR)
                plt.xlabel("Time (s)", color=TEXT_COLOR)
                plt.ylabel("Amplitude", color=TEXT_COLOR)
                plt.tight_layout()
                plt.show()
        except Exception as e:
            status_var.set(f"Plot error: {str(e)}")


def process_wav_file(file_path):
    try:
        clf = joblib.load(model_path)
        y, sr = librosa.load(file_path, sr=None)
        alpha = 0.97
        y_preemph = np.append(y[0], y[1:] - alpha * y[:-1])
        mfcc = librosa.feature.mfcc(y=y_preemph, sr=sr, n_mfcc=20)
        mfcc_cleaned = np.mean(mfcc.T, axis=0).reshape(1, -1)

        label = clf.predict(mfcc_cleaned)[0]
        probabilities = clf.predict_proba(mfcc_cleaned)[0]
        confidence = max(probabilities) * 100

        result = "Andi" if label == 0 else "Miro"
        color = "#1abc9c" if confidence > 85 else "#f39c12"  # Green for high confidence, orange for medium

        result_box.config(state=tk.NORMAL)
        result_box.insert(tk.END, f"{os.path.basename(file_path)}: ", "bold")
        result_box.insert(tk.END, f"{result} ", ("bold", "result"))
        result_box.insert(tk.END, f"({confidence:.2f}%)\n")
        result_box.tag_config("result", foreground=color)
        result_box.config(state=tk.DISABLED)
        result_box.see(tk.END)

        status_var.set(f"Processed: {os.path.basename(file_path)}")

    except Exception as e:
        status_var.set(f"Processing error: {str(e)}")


def process_selected_files():
    selected_files = get_selected_files()
    if not selected_files:
        status_var.set("Please select files to process")
        return

    result_box.config(state=tk.NORMAL)
    result_box.delete("1.0", tk.END)
    result_box.config(state=tk.DISABLED)

    for file_path in selected_files:
        process_wav_file(file_path)


# Recording functions
def record_audio():
    global is_recording, recorded_frames
    is_recording = True
    recorded_frames = []
    status_var.set("Recording...")
    record_btn.config(state=tk.DISABLED)
    stop_record_btn.config(state=tk.NORMAL)

    def callback(indata, frames, time, status):
        if is_recording:
            recorded_frames.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        while is_recording:
            sd.sleep(100)


def start_recording():
    threading.Thread(target=record_audio, daemon=True).start()


def stop_recording():
    global is_recording
    is_recording = False
    record_btn.config(state=tk.NORMAL)
    stop_record_btn.config(state=tk.DISABLED)

    if recorded_frames:
        audio = np.concatenate(recorded_frames, axis=0)
        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            title="Save Recording"
        )

        if save_path:
            write(save_path, samplerate, audio)
            if is_valid_wav(save_path):
                loaded_wav_files.append(save_path)
                listbox.insert(tk.END, os.path.basename(save_path))
                status_var.set(f"Saved: {os.path.basename(save_path)}")


# Live classification functions
def start_live_classification():
    live_led.set_color("#2ecc71")  # Green
    status_var.set("Live classification started")
    start_live_btn.config(state=tk.DISABLED)
    stop_live_btn.config(state=tk.NORMAL)

    # Start the worker in a new thread
    stop_event.clear()
    thread = threading.Thread(target=live_loop, daemon=True)
    thread.start()


def stop_live_classification():
    live_led.set_color("gray")
    status_var.set("Live classification stopped")
    start_live_btn.config(state=tk.NORMAL)
    stop_live_btn.config(state=tk.DISABLED)
    stop_event.set()


def live_loop():
    while not stop_event.is_set():
        try:
            ergraw = Gui_livedetection.live_val()
            erg = list(ergraw)
            # Update status with live results
            status_var.set(f"Live: {erg}")
        except Exception as e:
            status_var.set(f"Live error: {str(e)}")


# Info dialog
def show_info():
    messagebox.showinfo("How it Works",
                        "1. Load WAV files using 'Add Files' button\n"
                        "2. Select files and process them\n"
                        "3. Record audio using record buttons\n"
                        "4. Use live classification for real-time analysis\n\n"
                        "Results will show in the output box with confidence percentages.")


class VoiceClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Classifier Pro")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        self.loaded_files = []
        self.is_valid_wav = is_valid_wav
        # Dark theme colors
        self.bg_color = "#2d2d2d"
        self.text_color = "#ecf0f1"
        self.accent_color = "#3498db"

        self.root.configure(bg=self.bg_color)

        # Setup UI
        self.create_widgets()

        # Matplotlib figure for embedding plots
        self.figure = plt.Figure(figsize=(8, 4), dpi=100, facecolor=self.bg_color)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#3c3c3c')
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.text_color)
        self.ax.tick_params(colors=self.text_color)
        self.ax.xaxis.label.set_color(self.text_color)
        self.ax.yaxis.label.set_color(self.text_color)
        self.ax.title.set_color(self.text_color)

        # Embed matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_widgets(self):
        # Main frames
        control_frame = tk.Frame(self.root, bg=self.bg_color)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        list_frame = tk.Frame(self.root, bg=self.bg_color)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.plot_frame = tk.Frame(self.root, bg=self.bg_color)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        result_frame = tk.Frame(self.root, bg=self.bg_color)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Control buttons
        tk.Button(control_frame, text="Add Files", command=self.select_wav_files,
                  bg=self.accent_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Plot Selected", command=self.plot_selected_files,
                  bg=self.accent_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Classify Selected", command=self.process_selected_files,
                  bg=self.accent_color, fg=self.text_color).pack(side=tk.LEFT, padx=5)

        # File list with scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set,
                                  bg="#3c3c3c", fg=self.text_color, selectbackground=self.accent_color)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Results display
        result_scroll = tk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_box = tk.Text(result_frame, bg="#2c3e50", fg=self.text_color,
                                  yscrollcommand=result_scroll.set, wrap=tk.WORD)
        self.result_box.pack(fill=tk.BOTH, expand=True)
        result_scroll.config(command=self.result_box.yview)

    def select_wav_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select WAV files",
            filetypes=[("WAV files", "*.wav")]
        )

        if file_paths:
            for file_path in file_paths:
                if file_path not in self.loaded_files and self.is_valid_wav(file_path):
                    self.loaded_files.append(file_path)
                    self.listbox.insert(tk.END, os.path.basename(file_path))

    def plot_selected_files(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No selection", "Please select files to plot")
            return

        self.ax.clear()

        for i in selected_indices:
            file_path = self.loaded_files[i]
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
                    self.ax.plot(times, waveform, label=os.path.basename(file_path))

            except Exception as e:
                messagebox.showerror("Error", f"Could not plot {os.path.basename(file_path)}: {str(e)}")

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.canvas.draw()

    def process_selected_files(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No selection", "Please select files to classify")
            return

        self.result_box.delete(1.0, tk.END)

        for i in selected_indices:
            file_path = self.loaded_files[i]
            self.classify_audio(file_path)

    def classify_audio(self, file_path):
        try:
            clf = joblib.load(model_path)
            y, sr = librosa.load(file_path, sr=None)
            alpha = 0.97
            y_preemph = np.append(y[0], y[1:] - alpha * y[:-1])
            mfcc = librosa.feature.mfcc(y=y_preemph, sr=sr, n_mfcc=20)
            mfcc_cleaned = np.mean(mfcc.T, axis=0).reshape(1, -1)

            label = clf.predict(mfcc_cleaned)[0]
            probabilities = clf.predict_proba(mfcc_cleaned)[0]
            confidence = max(probabilities) * 100

            result = "Andi" if label == 0 else "Miro"
            color = "#1abc9c" if confidence > 85 else "#f39c12"

            self.result_box.insert(tk.END, f"{os.path.basename(file_path)}: ", "bold")
            self.result_box.insert(tk.END, f"{result} ", ("bold", "result"))
            self.result_box.insert(tk.END, f"({confidence:.2f}%)\n")
            self.result_box.tag_config("result", foreground=color)

        except Exception as e:
            self.result_box.insert(tk.END, f"Error processing {os.path.basename(file_path)}: {str(e)}\n")


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceClassifierApp(root)
    root.mainloop()