import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import wave
import os
import sys
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import Gui_livedetection
from PIL import Image, ImageTk


# Resource path handling
def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)


model_path = get_resource_path("voice_classifier_speech_final_no_noise_reduce2.0_20_pre_finalfcc.joblib")

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


class LEDIndicator(tk.Canvas):
    """Custom LED indicator widget"""

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


class VoiceClassifierApp:
    """Main application class for Voice Classifier Pro"""

    def __init__(self, root):
        self.root = root
        self.root.title("Voice Classifier Pro")
        self.root.geometry("1000x900")
        self.root.minsize(1100, 900)
        self.root.configure(bg=BG_COLOR)

        # Initialize variables
        self.loaded_wav_files = []
        self.is_recording = False
        self.recorded_frames = []
        self.samplerate = 44100
        self.stop_event = threading.Event()

        # Load model
        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model: {str(e)}")
            self.model_loaded = False

        # Setup UI
        self.create_widgets()
        self.create_menu()

        # Initialize matplotlib figure
        self.figure = plt.Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(ENTRY_COLOR)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)
        self.ax.tick_params(colors=TEXT_COLOR)
        self.ax.xaxis.label.set_color(TEXT_COLOR)
        self.ax.yaxis.label.set_color(TEXT_COLOR)
        self.ax.title.set_color(TEXT_COLOR)

        # Embed matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure styles
        self.configure_styles()

    def configure_styles(self):
        """Configure ttk styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure frame style
        style.configure('TFrame', background=BG_COLOR)

        # Configure label style
        style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))

        # Configure button style
        style.configure('TButton', background=BUTTON_COLOR, foreground=TEXT_COLOR,
                        font=('Segoe UI', 9, 'bold'), borderwidth=1, relief='flat')

        style.map('TButton', background=[('active', SECONDARY_COLOR), ('pressed', ACCENT_COLOR)],
                  foreground=[('active', TEXT_COLOR), ('pressed', TEXT_COLOR)])

        # Configure listbox style
        style.configure('Listbox', background=LISTBOX_COLOR, foreground=TEXT_COLOR,
                        selectbackground=ACCENT_COLOR, font=('Segoe UI', 9),
                        borderwidth=0, highlightthickness=0)

        # Configure scrollbar style
        style.configure('Vertical.TScrollbar', background=BG_COLOR, troughcolor=BG_COLOR,
                        bordercolor=BG_COLOR, arrowcolor=TEXT_COLOR, gripcount=0)

    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=BG_COLOR, fg=TEXT_COLOR)
        file_menu.add_command(label="Add Files", command=self.select_wav_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=BG_COLOR, fg=TEXT_COLOR)
        help_menu.add_command(label="How It Works", command=self.show_info)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Create main frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (file list)
        left_frame = ttk.LabelFrame(main_frame, text="Audio Files")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Right panel (controls and results)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # File list with scrollbar
        list_scroll = ttk.Scrollbar(left_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            left_frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=list_scroll.set,
            bg=LISTBOX_COLOR,
            fg=TEXT_COLOR,
            selectbackground=ACCENT_COLOR,
            font=('Segoe UI', 9)
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.listbox.yview)

        # File control buttons
        file_btn_frame = ttk.Frame(left_frame)
        file_btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_btn_frame, text="Add Files", command=self.select_wav_file).pack(side=tk.LEFT, padx=2, fill=tk.X,
                                                                                        expand=True)
        ttk.Button(file_btn_frame, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=2,
                                                                                              fill=tk.X, expand=True)
        ttk.Button(file_btn_frame, text="Clear All", command=self.clear_files).pack(side=tk.LEFT, padx=2, fill=tk.X,
                                                                                    expand=True)

        # Plot and process buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(control_frame, text="Plot Selected", command=self.plot_selected_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Classify Selected", command=self.process_selected_files).pack(side=tk.LEFT,
                                                                                                      padx=2)

        # Recording controls
        record_frame = ttk.LabelFrame(right_frame, text="Recording")
        record_frame.pack(fill=tk.X, pady=5)

        record_btn_frame = ttk.Frame(record_frame)
        record_btn_frame.pack(fill=tk.X, pady=5)

        self.record_btn = ttk.Button(record_btn_frame, text="Record", command=self.start_recording)
        self.record_btn.pack(side=tk.LEFT, padx=2)

        self.stop_record_btn = ttk.Button(record_btn_frame, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_record_btn.pack(side=tk.LEFT, padx=2)

        # Live classification controls
        live_frame = ttk.LabelFrame(right_frame, text="Live Classification")
        live_frame.pack(fill=tk.X, pady=5)

        live_btn_frame = ttk.Frame(live_frame)
        live_btn_frame.pack(fill=tk.X, pady=5)

        self.start_live_btn = ttk.Button(live_btn_frame, text="Start Live", command=self.start_live_classification)
        self.start_live_btn.pack(side=tk.LEFT, padx=2)

        self.stop_live_btn = ttk.Button(live_btn_frame, text="Stop", command=self.stop_live_classification,
                                        state=tk.DISABLED)
        self.stop_live_btn.pack(side=tk.LEFT, padx=2)

        self.live_led = LEDIndicator(live_btn_frame, size=20)
        self.live_led.pack(side=tk.LEFT, padx=10)

        # Plot frame
        self.plot_frame = ttk.LabelFrame(right_frame, text="Waveform Visualization")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Results frame
        result_frame = ttk.LabelFrame(right_frame, text="Classification Results")
        result_frame.pack(fill=tk.BOTH, expand=False, pady=5)

        result_scroll = ttk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_box = tk.Text(
            result_frame,
            bg=RESULT_BOX_COLOR,
            fg=TEXT_COLOR,
            yscrollcommand=result_scroll.set,
            wrap=tk.WORD,
            font=('Segoe UI', 9),
            height=8
        )
        self.result_box.pack(fill=tk.BOTH, expand=True)
        result_scroll.config(command=self.result_box.yview)

        # Status bar
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5)

    def is_valid_wav(self, file_path):
        """Check if a file is a valid WAV file"""
        try:
            with wave.open(file_path, 'rb') as wf:
                wf.getparams()
            return True
        except wave.Error:
            return False

    def select_wav_file(self):
        """Select and load WAV files"""
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
            if file_path not in self.loaded_wav_files and self.is_valid_wav(file_path):
                self.loaded_wav_files.append(file_path)
                self.listbox.insert(tk.END, os.path.basename(file_path))
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")

    def remove_selected(self):
        """Remove selected files from the list"""
        selected_indices = self.listbox.curselection()
        for i in selected_indices[::-1]:
            self.listbox.delete(i)
            del self.loaded_wav_files[i]

    def clear_files(self):
        """Clear all files from the list"""
        self.listbox.delete(0, tk.END)
        self.loaded_wav_files = []

    def get_selected_files(self):
        """Get paths of selected files"""
        selected_indices = self.listbox.curselection()
        return [self.loaded_wav_files[i] for i in selected_indices]

    def plot_selected_files(self):
        """Plot waveforms of selected files with grid and axis labels"""
        selected_files = self.get_selected_files()
        if not selected_files:
            self.status_var.set("Please select files to plot")
            return

        self.ax.clear()

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
                    self.ax.plot(times, waveform, label=os.path.basename(file_path))

            except Exception as e:
                self.status_var.set(f"Plot error: {str(e)}")

        # Add grid and axis labels
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        self.ax.set_xlabel("Time (s)", color=TEXT_COLOR)
        self.ax.set_ylabel("Amplitude", color=TEXT_COLOR)
        self.ax.set_title("Audio Waveform", color=TEXT_COLOR, pad=20)

        # Customize grid appearance
        self.ax.grid(color=TEXT_COLOR, alpha=0.3)

        # Customize ticks
        self.ax.tick_params(axis='both', which='both', colors=TEXT_COLOR)

        # Add legend with custom styling
        legend = self.ax.legend(facecolor=ENTRY_COLOR, labelcolor=TEXT_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        self.figure.tight_layout()
        self.canvas.draw()
        self.status_var.set(f"Plotted {len(selected_files)} file(s)")

    def process_selected_files(self):
        """Classify all selected files"""
        if not self.model_loaded:
            self.status_var.set("Model not loaded - cannot classify")
            return

        selected_files = self.get_selected_files()
        if not selected_files:
            self.status_var.set("Please select files to process")
            return

        self.result_box.config(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        self.result_box.config(state=tk.DISABLED)

        for file_path in selected_files:
            self.process_wav_file(file_path)

    def record_audio(self):
        """Audio recording thread function"""
        self.is_recording = True
        self.recorded_frames = []
        self.status_var.set("Recording...")
        self.record_btn.config(state=tk.DISABLED)
        self.stop_record_btn.config(state=tk.NORMAL)

        def callback(indata, frames, time, status):
            if self.is_recording:
                self.recorded_frames.append(indata.copy())

        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback):
            while self.is_recording:
                sd.sleep(100)

    def start_recording(self):
        """Start audio recording"""
        threading.Thread(target=self.record_audio, daemon=True).start()

    def stop_recording(self):
        """Stop audio recording and save file"""
        self.is_recording = False
        self.record_btn.config(state=tk.NORMAL)
        self.stop_record_btn.config(state=tk.DISABLED)

        if self.recorded_frames:
            audio = np.concatenate(self.recorded_frames, axis=0)
            save_path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")],
                title="Save Recording"
            )

            if save_path:
                write(save_path, self.samplerate, audio)
                if self.is_valid_wav(save_path):
                    self.loaded_wav_files.append(save_path)
                    self.listbox.insert(tk.END, os.path.basename(save_path))
                    self.status_var.set(f"Saved: {os.path.basename(save_path)}")

    def live_loop(self):
        """Live classification thread function"""
        while not self.stop_event.is_set():
            try:
                ergraw = Gui_livedetection.live_val()
                erg = list(ergraw)
                self.status_var.set(f"Live: {erg}")
            except Exception as e:
                self.status_var.set(f"Live error: {str(e)}")

    def start_live_classification(self):
        """Start live classification"""
        self.live_led.set_color("#2ecc71")  # Green
        self.status_var.set("Live classification started")
        self.start_live_btn.config(state=tk.DISABLED)
        self.stop_live_btn.config(state=tk.NORMAL)

        # Start the worker in a new thread
        self.stop_event.clear()
        threading.Thread(target=self.live_loop, daemon=True).start()

    def stop_live_classification(self):
        """Stop live classification"""
        self.live_led.set_color("gray")
        self.status_var.set("Live classification stopped")
        self.start_live_btn.config(state=tk.NORMAL)
        self.stop_live_btn.config(state=tk.DISABLED)
        self.stop_event.set()

    def show_info(self):
        """Show application instructions"""
        messagebox.showinfo("How it Works",
                            "1. Load WAV files using 'Add Files' button\n"
                            "2. Select files and process them\n"
                            "3. Record audio using record buttons\n"
                            "4. Use live classification for real-time analysis\n\n"
                            "Results will show in the output box with confidence percentages.")

    def show_about(self):
        """Show about dialog"""
        about_text = ("Voice Classifier Pro\n\n"
                      "Version 2.0\n"
                      "© 2023 Voice Analysis Systems\n"
                      "Developed with Python, Librosa and Scikit-learn")
        messagebox.showinfo("About Voice Classifier Pro", about_text)


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceClassifierApp(root)
    root.mainloop()