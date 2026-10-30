#!/usr/bin/env python3
"""
Personal Real-Time AI Live Call Assistant
Listens during live calls, retrieves verified facts from selected ChatGPT work profiles,
and displays concise, grounded suggestions to speak aloud.
"""

import os
import queue
import re
import sys
import threading
import time
import json
import zipfile
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import sounddevice as sd
import chromadb
from faster_whisper import WhisperModel
from openai import OpenAI

# --- APPLICATION SANDBOX DIRECTORIES ---
BASE_DIR = os.path.expanduser("~/.context_assistant")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
REVIEWS_DIR = os.path.join(BASE_DIR, "review_packets")

for directory in [BASE_DIR, PROFILES_DIR, VECTOR_DB_DIR, REVIEWS_DIR]:
    os.makedirs(directory, exist_ok=True)


class RealTimeAssistantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live Call Assistant -- Grounded ChatGPT Copilot")
        self.root.geometry("1060x720")
        self.root.minsize(920, 600)

        # Threading and Audio State
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.stream = None
        self.processing_thread = None
        self.current_samplerate = 16000
        self.transcript_history = []  # List of (timestamp, speaker, text)
        self.current_device_index = None

        # Storage & AI Connections
        self.chroma_client = None
        self.collection = None
        self.llm_client = None
        self.whisper_model = None
        self.llm_model_name = "llama3.1:8b"

        # Initialize Engines
        self.init_chromadb()
        self.init_llm_client()
        self.init_transcription_engine()

        # Build UI Layout
        self.build_ui()

        # Refresh Profiles & Audio Devices
        self.refresh_profiles_list()
        self.refresh_audio_devices()

        # Handle window close cleanly
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- INITIALIZATION ENGINES ---
    def init_chromadb(self):
        """Initializes persistent ChromaDB client for vector storage."""
        try:
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        except Exception as e:
            messagebox.showerror("Database Error", f"Could not initialize ChromaDB:\n{e}")

    def init_llm_client(self):
        """Initializes OpenAI client targeting local Ollama or cloud OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            # Cloud OpenAI fallback
            self.llm_client = OpenAI(api_key=api_key)
            self.llm_model_name = "gpt-4o-mini"
            print("[LLM Engine] Using OpenAI API (gpt-4o-mini)")
        else:
            # Local Ollama default
            self.llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.llm_model_name = "llama3.1:8b"
            # Verify if llama3.1 is available, otherwise resolve tag
            try:
                import urllib.request
                req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5)
                data = json.loads(req.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                if "llama3.1:8b" in models:
                    self.llm_model_name = "llama3.1:8b"
                elif "llama3.1" in models or any("llama3.1" in m for m in models):
                    self.llm_model_name = next(m for m in models if "llama3.1" in m)
                elif models:
                    self.llm_model_name = models[0]
                print(f"[LLM Engine] Connected to Ollama (Model: {self.llm_model_name})")
            except Exception:
                print(f"[LLM Engine] Ollama offline or stand-by; default: {self.llm_model_name}")

    def init_transcription_engine(self):
        """Initializes local faster-whisper speech-to-text model in background."""
        def load():
            try:
                # Fast int8 inference on Apple Silicon CPU / Metal
                self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
                print("[STT Engine] Faster-Whisper base.en loaded successfully.")
                self.root.after(0, self.update_status, "[SYSTEM READY]", "#30d158")
            except Exception as e:
                print(f"[STT Engine Error] Whisper initialization failed: {e}")
                self.root.after(0, self.update_status, "[STT FAILED]", "#ff453a")

        threading.Thread(target=load, daemon=True).start()

    # --- UI CONSTRUCTION ---
    def build_ui(self):
        """Builds the dark-mode interface."""
        style = ttk.Style()
        style.theme_use('clam')

        # Dropdown styling
        style.configure("TCombobox",
            fieldbackground="#2c2c2e",
            background="#3a3a3c",
            foreground="#ffffff",
            arrowcolor="#ffffff",
            darkcolor="#1c1c1e",
            lightcolor="#1c1c1e",
            bordercolor="#3a3a3c",
            padding=4
        )
        style.map("TCombobox",
            fieldbackground=[('readonly', '#2c2c2e')],
            selectbackground=[('readonly', '#2c2c2e')],
            selectforeground=[('readonly', '#ffffff')]
        )

        # Cross-platform TTK Button styles (guarantees clickability on macOS Aqua)
        style.configure("Start.TButton",
            background="#30d158",
            foreground="#000000",
            font=("SF Pro Text", 10, "bold"),
            padding=(12, 6),
            borderwidth=0,
            focuscolor="none"
        )
        style.map("Start.TButton",
            background=[('active', '#28a745'), ('pressed', '#1e7e34'), ('disabled', '#3a3a3c')],
            foreground=[('disabled', '#636366')]
        )

        style.configure("Stop.TButton",
            background="#ff453a",
            foreground="#ffffff",
            font=("SF Pro Text", 10, "bold"),
            padding=(12, 6),
            borderwidth=0,
            focuscolor="none"
        )
        style.map("Stop.TButton",
            background=[('active', '#d73328'), ('pressed', '#b52b20'), ('disabled', '#3a3a3c')],
            foreground=[('disabled', '#636366')]
        )

        style.configure("ActionBlue.TButton",
            background="#3a3a3c",
            foreground="#0a84ff",
            font=("SF Pro Text", 9, "bold"),
            padding=(8, 5),
            borderwidth=0,
            focuscolor="none"
        )
        style.map("ActionBlue.TButton",
            background=[('active', '#48484a'), ('pressed', '#2c2c2e')]
        )

        style.configure("ActionGreen.TButton",
            background="#3a3a3c",
            foreground="#30d158",
            font=("SF Pro Text", 9, "bold"),
            padding=(10, 5),
            borderwidth=0,
            focuscolor="none"
        )
        style.map("ActionGreen.TButton",
            background=[('active', '#48484a'), ('pressed', '#2c2c2e')]
        )

        style.configure("ActionRed.TButton",
            background="#3a3a3c",
            foreground="#ff453a",
            font=("SF Pro Text", 9, "bold"),
            padding=(8, 5),
            borderwidth=0,
            focuscolor="none"
        )
        style.map("ActionRed.TButton",
            background=[('active', '#48484a'), ('pressed', '#2c2c2e')]
        )

        style.configure("Primary.TButton",
            background="#0a84ff",
            foreground="#ffffff",
            font=("SF Pro Text", 10, "bold"),
            padding=(12, 6),
            borderwidth=0,
            focuscolor="none"
        )
        style.map("Primary.TButton",
            background=[('active', '#0071e3'), ('pressed', '#0058b0')]
        )

        self.root.configure(bg="#1c1c1e")

        # Top Control Strip: Profile + Audio Device + Actions
        top_bar = tk.Frame(self.root, bg="#2c2c2e", bd=0, padx=14, pady=10)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        # Profile Shelf
        lbl_prof = tk.Label(top_bar, text="Workspace Profile:", bg="#2c2c2e", fg="#aeaeaf", font=("SF Pro Text", 10, "bold"))
        lbl_prof.pack(side=tk.LEFT, padx=(0, 6))

        self.profile_var = tk.StringVar()
        self.profile_dropdown = ttk.Combobox(top_bar, textvariable=self.profile_var, state="readonly", width=18, font=("SF Pro Text", 10))
        self.profile_dropdown.pack(side=tk.LEFT, padx=(0, 8))
        self.profile_dropdown.bind("<<ComboboxSelected>>", self.on_profile_changed)

        btn_new_prof = ttk.Button(top_bar, text="+ New", style="ActionBlue.TButton", command=self.create_new_profile_dialog, cursor="pointinghand")
        btn_new_prof.pack(side=tk.LEFT, padx=(0, 4))

        btn_import = ttk.Button(top_bar, text="Import Context", style="ActionGreen.TButton", command=self.import_context_dialog, cursor="pointinghand")
        btn_import.pack(side=tk.LEFT, padx=(0, 4))

        btn_flush = ttk.Button(top_bar, text="Reset Memory", style="ActionRed.TButton", command=self.flush_context_confirm, cursor="pointinghand")
        btn_flush.pack(side=tk.LEFT, padx=(0, 16))

        # Audio Device Selector
        lbl_audio = tk.Label(top_bar, text="Audio Input:", bg="#2c2c2e", fg="#aeaeaf", font=("SF Pro Text", 10, "bold"))
        lbl_audio.pack(side=tk.LEFT, padx=(0, 6))

        self.audio_dev_var = tk.StringVar()
        self.audio_dropdown = ttk.Combobox(top_bar, textvariable=self.audio_dev_var, state="readonly", width=22, font=("SF Pro Text", 9))
        self.audio_dropdown.pack(side=tk.LEFT, padx=(0, 8))

        # Session Controls
        self.start_btn = ttk.Button(top_bar, text="Start Session", style="Start.TButton", command=self.start_session, cursor="pointinghand")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.stop_btn = ttk.Button(top_bar, text="Stop & Save", style="Stop.TButton", command=self.stop_session, state=tk.DISABLED, cursor="pointinghand")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.status_label = tk.Label(top_bar, text="[STARTING]", fg="#ff9f0a", bg="#2c2c2e", font=("SF Pro Text", 10, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=(6, 0))

        # Dual Panel Split View
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#1c1c1e", bd=0, sashwidth=6, sashrelief="flat")
        paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 12))

        # Left Panel: Transcript Stream
        left_frame = tk.LabelFrame(paned, text="  Live Call Dialogue Stream  ", bg="#1c1c1e", fg="#8e8e93", font=("SF Pro Text", 10, "bold"), bd=1, relief="solid")
        self.transcript_box = tk.Text(
            left_frame, bg="#000000", fg="#f2f2f7", insertbackground="#ffffff",
            wrap=tk.WORD, font=("SF Mono", 11), state=tk.DISABLED,
            bd=0, padx=12, pady=12, spacing1=4, spacing3=4
        )
        self.transcript_box.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        paned.add(left_frame, width=500)

        # Right Panel: Verified Grounded Suggestion Console
        right_frame = tk.LabelFrame(paned, text="  Verified Grounded Suggestion (Say Aloud)  ", bg="#1c1c1e", fg="#30d158", font=("SF Pro Text", 10, "bold"), bd=1, relief="solid")
        self.suggestion_box = tk.Text(
            right_frame, bg="#2c2c2e", fg="#30d158", insertbackground="#ffffff",
            wrap=tk.WORD, font=("SF Pro Text", 13, "bold"), state=tk.DISABLED,
            bd=0, padx=16, pady=16, spacing1=6, spacing3=6
        )
        self.suggestion_box.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        paned.add(right_frame, width=540)

        # Status Bar / Context Info at bottom
        self.bottom_bar = tk.Label(self.root, text="Select profile and click [Start Session].", bg="#1c1c1e", fg="#8e8e93", font=("SF Pro Text", 9), anchor="w", padx=14, pady=4)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def update_status(self, text: str, color: str = "#8e8e93"):
        """Refreshes status indicator safely."""
        self.status_label.config(text=text, fg=color)

    # --- PROFILE MANAGEMENT ---
    def refresh_profiles_list(self):
        """Scans profile directory and updates combobox selector."""
        subdirs = [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d)) and not d.startswith('.')]
        if not subdirs:
            default_path = os.path.join(PROFILES_DIR, "Default_Client")
            os.makedirs(default_path, exist_ok=True)
            subdirs = ["Default_Client"]

        subdirs.sort()
        self.profile_dropdown['values'] = subdirs
        current = self.profile_var.get()
        if not current or current not in subdirs:
            self.profile_var.set(subdirs[0])
            self.on_profile_changed(None)

    def on_profile_changed(self, event):
        """Switches active vector database collection when profile is switched."""
        active = self.profile_var.get()
        if not active:
            return
        safe_name = "prof_" + re.sub(r'[^a-zA-Z0-9_-]', '_', active).lower()[:50]
        try:
            self.collection = self.chroma_client.get_or_create_collection(name=safe_name)
            doc_count = self.collection.count()
            self.update_suggestion(
                f"Active Profile: [{active}]\n"
                f"Knowledge Memory: {doc_count} indexed context chunks.\n\n"
                f"When the caller speaks, verified answers based on this client profile will appear here."
            )
            self.bottom_bar.config(text=f"Profile: {active} | Indexed Fragments: {doc_count} | Ready.")
        except Exception as e:
            messagebox.showerror("Profile Selection Error", f"Failed to open collection for {active}:\n{e}")

    def create_new_profile_dialog(self):
        """Dialog to name and create a new isolated client profile."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New Profile")
        dialog.geometry("380x150")
        dialog.configure(bg="#2c2c2e")
        dialog.transient(self.root)
        dialog.grab_set()

        lbl = tk.Label(dialog, text="Enter Profile Name (e.g. Acme_Corp):", bg="#2c2c2e", fg="#ffffff", font=("SF Pro Text", 10))
        lbl.pack(pady=(16, 8))

        entry = tk.Entry(dialog, width=28, font=("SF Pro Text", 11), bg="#1c1c1e", fg="#ffffff", insertbackground="#ffffff")
        entry.pack(pady=4)
        entry.focus()

        def save():
            name = entry.get().strip()
            name_clean = re.sub(r'\s+', '_', name)
            if not name_clean:
                messagebox.showwarning("Invalid Name", "Please enter a valid profile name.")
                return
            prof_dir = os.path.join(PROFILES_DIR, name_clean)
            os.makedirs(prof_dir, exist_ok=True)
            self.refresh_profiles_list()
            self.profile_var.set(name_clean)
            self.on_profile_changed(None)
            dialog.destroy()

        btn = ttk.Button(dialog, text="Create Profile", style="Primary.TButton", command=save, cursor="pointinghand")
        btn.pack(pady=12)

    def flush_context_confirm(self):
        """Asks confirmation before erasing indexed fragments for the active profile."""
        active = self.profile_var.get()
        confirm = messagebox.askyesno(
            "Confirm Context Reset",
            f"Are you sure you want to completely erase all indexed knowledge fragments for [{active}]?\n\nThis action cannot be undone."
        )
        if confirm:
            try:
                safe_name = "prof_" + re.sub(r'[^a-zA-Z0-9_-]', '_', active).lower()[:50]
                self.chroma_client.delete_collection(name=safe_name)
                self.collection = self.chroma_client.get_or_create_collection(name=safe_name)
                self.update_suggestion(f"Context wiped for profile [{active}]. Total entries: 0.")
                self.bottom_bar.config(text=f"Profile: {active} | Indexed Fragments: 0 | Flushed.")
                messagebox.showinfo("Memory Reset", f"Profile [{active}] memory cleared successfully.")
            except Exception as e:
                messagebox.showerror("Reset Failed", f"Could not clear profile:\n{e}")

    # --- INGESTION & CHATGPT PARSING ENGINE ---
    def import_context_dialog(self):
        """Opens file dialog to import ChatGPT conversation export."""
        file_path = filedialog.askopenfilename(
            title="Select ChatGPT Export File",
            filetypes=[
                ("Supported Exports", "*.json *.zip *.md *.txt"),
                ("JSON Files", "*.json"),
                ("ZIP Archives", "*.zip"),
                ("Markdown / Text", "*.md *.txt"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return

        threading.Thread(target=self.process_and_index_file, args=(file_path,), daemon=True).start()

    def process_and_index_file(self, file_path: str):
        """Parses exported files, segments dialogue nodes, and loads into ChromaDB."""
        self.root.after(0, self.update_suggestion, "Parsing selected context file... Please wait.")
        extracted_chunks = []
        try:
            filename = os.path.basename(file_path).lower()

            # Scenario A: Full ChatGPT account ZIP archive
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as archive:
                    if 'conversations.json' in archive.namelist():
                        with archive.open('conversations.json') as f:
                            raw_data = json.load(f)
                            extracted_chunks.extend(self.parse_chatgpt_json_structure(raw_data))

            # Scenario B: Exported conversations.json
            elif filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    extracted_chunks.extend(self.parse_chatgpt_json_structure(raw_data))

            # Scenario C: Markdown or Plain Text notes/transcripts
            elif filename.endswith('.md') or filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    extracted_chunks = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 25]

            if not extracted_chunks:
                raise ValueError("No viable conversation messages or text paragraphs discovered in the file.")

            # Load into ChromaDB collection
            active = self.profile_var.get()
            timestamp = int(time.time())
            ids = [f"{active}_{timestamp}_{i}" for i in range(len(extracted_chunks))]
            self.collection.add(documents=extracted_chunks, ids=ids)

            count = self.collection.count()
            self.root.after(0, lambda: messagebox.showinfo(
                "Ingestion Complete",
                f"Successfully loaded {len(extracted_chunks)} context fragments into [{active}]!\nTotal knowledge base: {count} items."
            ))
            self.root.after(0, self.update_suggestion,
                f"Profile Context Refreshed: [{active}]\n"
                f"Total Indexed Knowledge Fragments: {count}\n\n"
                f"Assistant is primed and ready to support your live calls."
            )
            self.root.after(0, lambda: self.bottom_bar.config(text=f"Profile: {active} | Indexed Fragments: {count} | Ready."))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Parsing Error", f"Failed to ingest file:\n{e}"))
            self.root.after(0, self.update_suggestion, "Ready. Ingestion suspended.")

    def parse_chatgpt_json_structure(self, json_data) -> list:
        """Parses OpenAI's hierarchical conversation node trees into clean grounding strings."""
        parsed = []
        conversations = json_data if isinstance(json_data, list) else [json_data]

        for chat in conversations:
            title = chat.get('title', 'Historical Work Thread')
            mapping = chat.get('mapping', {})
            for node_id, node in mapping.items():
                message = node.get('message')
                if message and message.get('content') and message['content'].get('parts'):
                    role = message.get('author', {}).get('role', 'user')
                    parts = message['content']['parts']
                    text = " ".join([p for p in parts if isinstance(p, str)]).strip()
                    if text and len(text) > 15:
                        entry = f"[Thread: {title}] ({role}): {text}"
                        parsed.append(entry)

        return parsed

    # --- AUDIO DEVICE DISCOVERY & ROUTING ---
    def refresh_audio_devices(self):
        """Scans host devices and populates dropdown with BlackHole or default."""
        devices = sd.query_devices()
        options = ["Auto (Detect Loopback / Default)"]
        default_idx = 0
        bh_found = False

        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                is_bh = "BlackHole" in dev['name']
                label = f"[{idx}] {dev['name']}" + (" (Loopback)" if is_bh else "")
                options.append(label)
                if is_bh and not bh_found:
                    default_idx = len(options) - 1
                    bh_found = True

        self.audio_dropdown['values'] = options
        self.audio_dropdown.current(default_idx)

    def get_selected_device_index(self) -> int:
        """Resolves selected device index from dropdown."""
        choice = self.audio_dev_var.get()
        devices = sd.query_devices()

        if choice.startswith("Auto"):
            # Try to find BlackHole first
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0 and "BlackHole" in dev['name']:
                    return idx
            # Fallback to system default input
            def_in = sd.default.device[0]
            if def_in is not None and def_in >= 0:
                return def_in
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    return idx
            raise RuntimeError("No input audio devices found on this system.")

        # Parse index from label e.g. "[2] Micrófono..."
        match = re.match(r'\[(\d+)\]', choice)
        if match:
            return int(match.group(1))

        return sd.default.device[0]

    # --- LIVE AUDIO & TRANSCRIPTION PIPELINE ---
    def audio_callback(self, indata, frames, time_info, status):
        """Pushes raw incoming PCM frames into thread-safe buffer."""
        if status:
            print(f"[Audio Status] {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def start_session(self):
        """Hooks audio stream and starts background speech-to-text loop."""
        if self.collection is None or self.collection.count() == 0:
            ans = messagebox.askyesno(
                "Empty Knowledge Base",
                f"Profile [{self.profile_var.get()}] has 0 imported context documents.\n\n"
                "Would you like to start the session anyway (without work history)?"
            )
            if not ans:
                return

        if self.whisper_model is None:
            messagebox.showwarning("Engine Initializing", "Whisper STT engine is still loading. Please wait a few seconds.")
            return

        try:
            device_index = self.get_selected_device_index()
            dev_info = sd.query_devices(device_index)
            self.current_samplerate = int(dev_info.get('default_samplerate', 16000))
            self.current_device_index = device_index

            # Dynamic 250ms block size
            blocksize = int(self.current_samplerate * 0.25)
            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=self.current_samplerate,
                callback=self.audio_callback,
                blocksize=blocksize
            )
            self.stream.start()

            self.is_listening = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.update_status("[LISTENING]", "#30d158")
            self.bottom_bar.config(text=f"Listening via: {dev_info['name']} ({self.current_samplerate}Hz) | Real-time suggestions enabled.")

            # Clear session transcript history
            self.transcript_history.clear()

            # Start worker thread
            self.processing_thread = threading.Thread(target=self.process_audio_pipeline, daemon=True)
            self.processing_thread.start()

        except Exception as e:
            messagebox.showerror("Audio Launch Fault", f"Could not open audio stream on device [{device_index}]:\n{e}")
            self.stop_session()

    def stop_session(self):
        """Stops stream, exports review packet, and restores idle UI state."""
        self.is_listening = False
        if self.stream and self.stream.active:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("[STOPPED]", "#ff453a")
        self.bottom_bar.config(text="Session stopped. Review packet exported.")

        # Export post-call summary
        self.export_post_call_packet()

    def process_audio_pipeline(self):
        """Worker thread: buffers audio frames, performs VAD, and dispatches phrases."""
        buffer = np.zeros((0, 1), dtype=np.float32)
        silence_threshold = 0.015
        silence_frames = int(self.current_samplerate * 1.5)  # 1.5 sec silence threshold

        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.2)
                buffer = np.vstack((buffer, chunk))
            except queue.Empty:
                if len(buffer) > silence_frames:
                    energy = float(np.sqrt(np.mean(buffer**2)))
                    if energy < silence_threshold and len(buffer) > int(self.current_samplerate * 0.8):
                        # Phrase concluded: dispatch audio
                        phrase_audio = buffer.flatten()
                        buffer = np.zeros((0, 1), dtype=np.float32)
                        self.process_phrase(phrase_audio)
                    elif energy < silence_threshold:
                        # Empty silence reset
                        buffer = np.zeros((0, 1), dtype=np.float32)
                continue

    def process_phrase(self, audio_data: np.ndarray):
        """Resamples to 16kHz if needed, transcribes via Whisper, and triggers suggestions."""
        if len(audio_data) < 4000:
            return

        # Resample to 16000 Hz if hardware uses 44.1k/48k
        if self.current_samplerate != 16000:
            target_len = int(len(audio_data) * 16000 / self.current_samplerate)
            audio_16k = np.interp(
                np.linspace(0, len(audio_data), target_len, endpoint=False),
                np.arange(len(audio_data)),
                audio_data
            ).astype(np.float32)
        else:
            audio_16k = audio_data.astype(np.float32)

        try:
            segments, _ = self.whisper_model.transcribe(audio_16k, beam_size=5)
            text = " ".join([seg.text for seg in segments]).strip()
            if text and len(text) > 3:
                speaker = "Participant"
                self.transcript_history.append((time.strftime("%H:%M:%S"), speaker, text))
                self.root.after(0, self.append_transcript, text, speaker)
                # Dispatch suggestion generation on worker thread
                threading.Thread(target=self.generate_contextual_suggestion, args=(text,), daemon=True).start()
        except Exception as e:
            print(f"[Transcription Error]: {e}", file=sys.stderr)

    def append_transcript(self, text: str, speaker: str):
        """Appends dialogue safely to transcript view."""
        self.transcript_box.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.transcript_box.insert(tk.END, f"[{timestamp}] {speaker}:\n{text}\n\n")
        self.transcript_box.see(tk.END)
        self.transcript_box.config(state=tk.DISABLED)

    def update_suggestion(self, text: str):
        """Refreshes the suggestion box with formatted text."""
        self.suggestion_box.config(state=tk.NORMAL)
        self.suggestion_box.delete("1.0", tk.END)
        self.suggestion_box.insert(tk.END, text)
        self.suggestion_box.config(state=tk.DISABLED)

    # --- RETRIEVAL & GROUNDED SUGGESTION ENGINE ---
    def generate_contextual_suggestion(self, incoming_text: str):
        """Runs vector search against active profile and prompts LLM with strict grounding."""
        context_str = "No specific history entries discovered for this profile target."

        if self.collection and self.collection.count() > 0:
            try:
                results = self.collection.query(query_texts=[incoming_text], n_results=3)
                if results.get('documents') and results['documents'][0]:
                    context_str = "\n".join(results['documents'][0])
            except Exception as e:
                print(f"[ChromaDB Query Error]: {e}")

        system_prompt = (
            "You are a real-time conversational co-pilot for a live work call.\n"
            "Formulate suggestions strictly based upon the provided work history context below.\n\n"
            f"=== VERIFIED CLIENT / WORK HISTORY ===\n{context_str}\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. Output a concise 1-2 sentence response the user can say aloud immediately.\n"
            "2. If the historical context does NOT contain the answer, answer EXACTLY:\n"
            "   'I don't have that documented in our prior work history, but let me check and follow up.'\n"
            "3. Do NOT invent dates, financial figures, commitments, or approvals.\n"
            "4. Append a useful follow-up question in brackets [like this] when appropriate.\n"
            "5. Speak directly in first-person ('We agreed...', 'Our deadline is...')."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Participant said: '{incoming_text}'"}
                ],
                temperature=0.0,
                max_tokens=150
            )
            suggestion = response.choices[0].message.content.strip()
            self.root.after(0, self.update_suggestion, suggestion)
        except Exception as e:
            err_msg = (
                f"LLM Connection Error:\n"
                f"Ensure Ollama is running ('ollama run llama3.1') or set OPENAI_API_KEY.\n\nDetails: {e}"
            )
            self.root.after(0, self.update_suggestion, err_msg)

    # --- POST-CALL REVIEW PACKET EXPORT ---
    def export_post_call_packet(self):
        """Exports structured Markdown packet formatted for direct ChatGPT Plus context refresh."""
        if not self.transcript_history:
            return

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(REVIEWS_DIR, f"review_packet_{timestamp_str}.md")
        active = self.profile_var.get()

        formatted_dialogue = ""
        for t_stamp, speaker, phrase in self.transcript_history:
            formatted_dialogue += f"[{t_stamp}] {speaker}: {phrase}\n"

        try:
            with open(export_path, "w", encoding='utf-8') as f:
                f.write("# SYSTEM SYNC PACKET: LIVE CALL PROFILE REFRESH\n")
                f.write(f"Generated On: {time.strftime('%Y-%m-%d %H:%M:%S')} | Target Profile: {active}\n")
                f.write("=" * 70 + "\n\n")

                f.write("## INSTRUCTIONS FOR CHATGPT CONTEXT REFRESH\n")
                f.write("Copy and paste the entire block container below directly into your primary workspace chat thread:\n\n")

                f.write("```text\n")
                f.write("UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT\n")
                f.write(f"TARGET_PROFILE_CONTEXT: {active}\n")
                f.write(f"SYNC_DATE: {time.strftime('%Y-%m-%d')}\n\n")

                f.write("=== RAW CAPTURED SYNC DIALOGUE ===\n")
                f.write(formatted_dialogue)
                f.write("\n")

                f.write("=== REQUIRED INFERENCE ACTIONS ===\n")
                f.write("Analyze the dialogue block above and update your persistent internal system memory for this profile:\n")
                f.write("1. Identify any newly authorized agreements, project deadlines, or explicit change requests.\n")
                f.write("2. Note any missing or requested facts that require out-of-band operational follow-up.\n")
                f.write("3. Refuse to delete past historical core assumptions unless explicit contradictions appear in the text.\n")
                f.write("4. Respond with a concise single-sentence confirmation stating: 'Workspace memory array successfully synced.'\n")
                f.write("```\n\n")

                f.write("## APPENDIX: LOCAL AUDIT TRANSCRIPT LOG\n")
                f.write(f"Profile: {active}\n\n")
                for t_stamp, speaker, phrase in self.transcript_history:
                    f.write(f"- [{t_stamp}] {speaker}: {phrase}\n")

            print(f"[Export System] Saved review packet to: {export_path}")
            messagebox.showinfo(
                "Session Exported",
                f"A ChatGPT-ready post-call review packet has been saved to:\n\n{export_path}"
            )
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not write review packet to disk:\n{e}")

    def on_close(self):
        """Cleanly stops audio streams and exits application."""
        self.is_listening = False
        if hasattr(self, 'stream') and self.stream and self.stream.active:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeAssistantApp(root)
    # Ensure window is elevated and receives native mouse/keyboard focus on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()
    root.mainloop()
