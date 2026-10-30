#!/usr/bin/env python3
"""
Personal Real-Time AI Live Call Assistant — Aurora UI Edition
A modern, declarative desktop copilot built with Aurora (Rust/Skia/Taffy engine).

Design & Capabilities:
- Full LLM Provider & Model Configuration (Ollama local/remote URL, OpenAI ChatGPT, Custom OpenAI-Compatible).
- Dynamic remote model discovery from /v1/models & /api/tags endpoints.
- Harmonic pastel color palette (Periwinkle, Mint, Coral, Sky Blue, and Slate).
- 100% SVG vector line icons (no emojis, unicode glyphs, or rare non-ascii fonts).
- Custom modern application logo with pastel gradient squircle.
- Multi-page navigation (Live Copilot, Context & Profiles, Review Packets, Settings).
- Real-time audio loopback (BlackHole) & mic capture with energy VAD (1.5s silence).
- Dynamic sample rate resampling to 16,000 Hz single-channel float32.
- Local speech-to-text with Faster-Whisper (base.en, int8 CPU/Metal).
- Strict zero-hallucination grounding with explicit refusal invariant.
- Post-call review packets with 1-click ChatGPT memory sync prompt.
"""

import os
import queue
import re
import subprocess
import sys
import threading
import time
import json
import warnings
import zipfile
import urllib.request
from typing import List, Tuple, Dict, Any, Optional

# Suppress benign multiprocessing resource_tracker cleanup warning on shutdown
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

import numpy as np
import sounddevice as sd
import chromadb
from faster_whisper import WhisperModel
from openai import OpenAI

from aurora import (
    App, Signal, Element, Theme,
    column, row, box, card, text, badge, select, spinner,
    scroll, spacer, divider, dialog, toast, text_input, textarea, dialogs, pages, image
)

from audio_devices import multi_output_devices, resolve_meeting_device, short_device_label
from call_export import (
    FORMATS,
    KIND_CONVERSATION,
    KIND_RESUME,
    default_filename,
    ensure_extension,
    profile_from_packet,
    render_document,
    turns_from_packet,
)
from speech_audio import is_cross_talk_echo, prepare_speech_audio, take_speech_phrase, to_mono, transcribe_speech
from ui_icons import get_svg_icon, get_logo_svg

# --- APPLICATION SANDBOX DIRECTORIES ---
BASE_DIR = os.path.expanduser("~/.context_assistant")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
REVIEWS_DIR = os.path.join(BASE_DIR, "review_packets")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

for d in [BASE_DIR, PROFILES_DIR, VECTOR_DB_DIR, REVIEWS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- APPLICATION CONFIGURATION DEFAULTS ---
DEFAULT_SYSTEM_PROMPT = (
    "You are an executive real-time conversational co-pilot for a live work call.\n"
    "Your mission is to provide the user with a concise, highly factual response to speak aloud, strictly anchored in the verified work history below.\n\n"
    "=== VERIFIED CLIENT / WORK HISTORY ===\n"
    "{context}\n\n"
    "CORE OPERATIONAL RULES:\n"
    "1. DIRECT MEMORY ANCHORING: Formulate your answer by explicitly citing the exact agreed-upon facts, figures, dates, deliverables, or decisions found in the VERIFIED CLIENT / WORK HISTORY above (e.g., 'We agreed to a budget cap of $45,000...', 'Our target delivery date is November 15, 2026...').\n"
    "2. SPEAK NATURALLY IN FIRST PERSON: Respond directly as the user speaking aloud to the call participant ('We agreed...', 'Our documented timeline is...', 'In our previous discussion...').\n"
    "3. STRICT ZERO HALLUCINATION: If the verified history does NOT contain the specific fact or answer to the participant's question, you must respond EXACTLY with:\n"
    "   \"I don't have that documented in our prior work history, but let me check and follow up.\"\n"
    "   Never invent figures, dates, agreements, or commitments.\n"
    "4. CONTEXTUAL FOLLOW-UP: Append a practical follow-up question in brackets [like this] when appropriate to keep the meeting productive.\n"
    "5. MAXIMUM CONCISENESS: Limit your response to 1-2 clear, spoken sentences. Avoid conversational filler or meta-commentary outside the spoken response."
)


ENGLISH_MODELS = [
    ("small.en", "Small English (Fast & Accurate)"),
    ("distil-small.en", "Distil Small (Fastest)"),
    ("base.en", "Base English (Light)"),
    ("distil-medium.en", "Distil Medium (High Fidelity)"),
    ("medium.en", "Medium English (Max Quality)"),
]

MULTILINGUAL_MODELS = [
    ("small", "Small Multilingual (Balanced)"),
    ("base", "Base Multilingual (Fastest)"),
    ("medium", "Medium Multilingual (High Quality)"),
    ("distil-large-v3", "Distil Large v3 (Fast Max Quality)"),
    ("large-v3", "Large v3 (Maximum Accuracy)"),
]

LANGUAGE_OPTIONS = [
    ("en", "English"),
    ("es", "Spanish (Espanol)"),
    ("fr", "French (Francais)"),
    ("de", "German (Deutsch)"),
    ("it", "Italian (Italiano)"),
    ("pt", "Portuguese (Portugues)"),
    ("nl", "Dutch (Nederlands)"),
    ("ru", "Russian (Russkiy)"),
    ("zh", "Chinese (Zhongwen)"),
    ("ja", "Japanese (Nihongo)"),
    ("ko", "Korean (Hangul)"),
    ("auto", "Auto-Detect Language"),
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "ollama",
    "ollama_url": "http://localhost:11434",
    "openai_api_key": "",
    "custom_url": "http://localhost:1234/v1",
    "custom_api_key": "",
    "selected_model": "llama3.1:8b",
    "stt_language": "en",
    "whisper_model": "small.en",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.0,
    "max_tokens": 150,
    "silence_threshold": 0.015,
    "mic_device": "auto",
    "meeting_device": "auto",
}


# --- PASTEL DESIGN SYSTEM THEMES ---
PASTEL_DARK = Theme(
    colors={
        "background": "#0F1219",
        "surface": "#181D2A",
        "surface_hover": "#22293B",
        "surface_pressed": "#2B344B",
        "border": "#283147",
        "primary": "#A5B4FC",         # Soft pastel periwinkle
        "primary_hover": "#B4C1FD",
        "primary_pressed": "#93A3F8",
        "accent": "#6EE7B7",          # Soft pastel mint
        "success": "#86EFAC",         # Soft pastel emerald
        "danger": "#FDA4AF",          # Soft pastel coral/rose
        "muted": "#94A3B8",           # Slate muted
        "text": "#F8FAFC",            # Luminous off-white
        "glow": "#A5B4FC26",
        "shadow": "#00000060",
    },
    radius={"sm": 8.0, "md": 12.0, "lg": 16.0, "full": 9999.0}
)

PASTEL_LIGHT = Theme(
    colors={
        "background": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface_hover": "#F1F5F9",
        "surface_pressed": "#E2E8F0",
        "border": "#E2E8F0",
        "primary": "#6366F1",
        "primary_hover": "#4F46E5",
        "primary_pressed": "#4338CA",
        "accent": "#059669",
        "success": "#16A34A",
        "danger": "#E11D48",
        "muted": "#64748B",
        "text": "#0F172A",
        "glow": "#6366F126",
        "shadow": "#00000015",
    },
    radius={"sm": 8.0, "md": 12.0, "lg": 16.0, "full": 9999.0}
)

# Compatibility aliases
MIDNIGHT = PASTEL_DARK
DAYLIGHT = PASTEL_LIGHT


# --- REUSABLE UI PRIMITIVES ---

def pill(label: str, color: str = "#A5B4FC", *, solid: bool = False, dot_color: Optional[str] = None) -> Element:
    """Renders a sleek capsule pill with soft pastel colors (no emojis)."""
    bg_color = color if solid else (color + "20" if color.startswith("#") else "$surface_pressed")
    fg_color = "#0F1219" if solid else color

    items = []
    if dot_color:
        items.append(box(width=6, height=6, radius="$full", bg=dot_color))
    items.append(text(label, size=11, weight=700, color=fg_color))

    return row(gap=5, align="center", padding=(4, 10), radius="$full", bg=bg_color)[items]


def icon_box(icon_name: str, color: str = "#A5B4FC", size: int = 28, icon_size: int = 14) -> Element:
    """Renders a soft pastel rounded icon container."""
    return box(
        width=size,
        height=size,
        radius=8,
        align="center",
        justify="center",
        bg=f"{color}20",
    )[
        image(get_svg_icon(icon_name, color, icon_size), width=icon_size, height=icon_size)
    ]


def action_btn(
    label: str,
    icon: Optional[str] = None,
    on_click=None,
    variant: str = "soft",
    key: Optional[str] = None,
    opacity: float = 1.0,
    disabled: bool = False
) -> Element:
    """Renders an interactive button with vector SVG icon and refined pastel styling."""
    styles = {
        "primary": {"bg": "#A5B4FC", "bg_hover": "#B4C1FD", "fg": "#0F1219", "border": None},
        "danger":  {"bg": "#FDA4AF", "bg_hover": "#FECDD3", "fg": "#0F1219", "border": None},
        "mint":    {"bg": "#6EE7B7", "bg_hover": "#86EFAC", "fg": "#0F1219", "border": None},
        "soft":    {"bg": "$surface_hover", "bg_hover": "$surface_pressed", "fg": "$text", "border": "$border"},
        "ghost":   {"bg": "#00000000", "bg_hover": "$surface_hover", "fg": "$muted", "border": None},
    }
    st = styles.get(variant, styles["soft"])
    icon_color = ctrl.current_theme().resolve(st["fg"])

    children = []
    if icon:
        children.append(image(get_svg_icon(icon, icon_color, 14), width=14, height=14))
    children.append(text(label, size=13, weight=650, color=st["fg"]))

    extra = {}
    if st["border"]:
        extra["border_width"] = 1
        extra["border_color"] = st["border"]

    click_handler = None if disabled else on_click

    return row(
        key=key,
        align="center",
        gap=8,
        padding=(8, 14),
        radius=10,
        bg=st["bg"],
        bg_hover=st["bg_hover"] if not disabled else st["bg"],
        on_click=click_handler,
        opacity=opacity,
        **extra
    )[children]


class AuroraAssistantController:
    """State manager and background processor for the Aurora Assistant."""

    def __init__(self):
        # Navigation & Theme State
        self.active_page = Signal("copilot")  # "copilot", "profiles", "packets", "settings"
        self.current_theme = Signal(PASTEL_DARK)

        # Reactive Profile & Audio Device State
        self.profiles_options = Signal([("Default_Client", "Default_Client")])
        self.selected_profile = Signal("Default_Client")
        self.mic_options = Signal([("auto", "System default microphone")])
        self.meeting_options = Signal([("auto", "BlackHole if available")])
        self.selected_mic = Signal("auto")
        self.selected_meeting = Signal("auto")

        # Session & Status Signals
        self.is_listening = Signal(False)
        self.status_text = Signal("SYSTEM READY")
        self.status_color = Signal("#86EFAC")  # Pastel emerald
        self.doc_count = Signal(0)
        self.active_device_label = Signal("Auto (Loopback / Default)")
        self.hardware_sr_label = Signal("16000 Hz")

        # LLM Provider & Model Discovery State
        self.llm_provider = Signal(DEFAULT_CONFIG["provider"])  # "ollama", "openai", "custom"
        self.ollama_url_input = Signal(DEFAULT_CONFIG["ollama_url"])
        self.openai_key_input = Signal(DEFAULT_CONFIG["openai_api_key"])
        self.custom_url_input = Signal(DEFAULT_CONFIG["custom_url"])
        self.custom_key_input = Signal(DEFAULT_CONFIG["custom_api_key"])
        self.available_models = Signal([(DEFAULT_CONFIG["selected_model"], DEFAULT_CONFIG["selected_model"])])
        self.selected_model = Signal(DEFAULT_CONFIG["selected_model"])
        # Speech-to-Text Language & Adapted Model Selection
        self.stt_language = Signal(DEFAULT_CONFIG["stt_language"])
        self.language_options = LANGUAGE_OPTIONS
        init_choices = ENGLISH_MODELS if self.stt_language() == "en" else MULTILINGUAL_MODELS
        self.whisper_model_choices = Signal(init_choices)
        self.whisper_model_option = Signal(DEFAULT_CONFIG["whisper_model"])
        self.model_status_note = Signal("Connecting to model endpoint...")
        self.engine_model_label = Signal("Connecting to Ollama...")

        # Prompt & Generation Parameters State
        self.system_prompt = Signal(DEFAULT_CONFIG["system_prompt"])
        self.temperature = Signal(str(DEFAULT_CONFIG["temperature"]))
        self.max_tokens = Signal(str(DEFAULT_CONFIG["max_tokens"]))
        self.silence_threshold = Signal(str(DEFAULT_CONFIG["silence_threshold"]))

        # Dialogue & Suggestions State
        self.transcripts = Signal([])  # list of dicts: {"time": str, "speaker": str, "text": str}
        self.suggestion_text = Signal("Assistant is primed. Click [Start Session] to begin real-time grounding.")
        self.retrieved_sources = Signal([])  # list of top-3 cited ChromaDB chunks
        self.bottom_info = Signal("Ready. Live-call audio loopback and microphone ready to monitor.")
        self.copy_btn_state = Signal(("Copy", "copy"))  # (label, icon_name)

        # In-App Memory Browser & Review Packets State
        self.profile_chunks = Signal([])  # list of documents in active profile
        self.review_packets_list = Signal([])  # list of review packet file info dicts
        self.selected_packet_content = Signal("")
        self.selected_packet_name = Signal("")
        self.packet_copy_state = Signal(("Copy ChatGPT Sync Prompt", "copy"))

        # Modals & Notifications
        self.show_new_profile_dialog = Signal(False)
        self.new_profile_name = Signal("")
        self.show_reset_dialog = Signal(False)
        self.show_export_dialog = Signal(False)
        self.export_kind = Signal(KIND_CONVERSATION)
        self.export_format = Signal("md")
        self.export_source = Signal("live")
        self.is_closing = Signal(False)
        self.closing_status = Signal("Preparing shutdown...")
        self.toast_msg = Signal(None)
        self.toast_tone = Signal("$accent")

        # Application Reference (for app.quit())
        self.app = None

        # Backend AI & Audio State
        self.chroma_client = None
        self.collection = None
        self.llm_client = None
        self.llm_model_name = "llama3.1:8b"
        self.whisper_model = None
        self._whisper_lock = threading.Lock()

        self.audio_queue = queue.Queue()
        self.stream = None
        self.streams = []
        self._has_meeting_stream = False
        self.processing_thread = None
        self.current_samplerate = 16000
        self.transcript_history = []  # list of (timestamp, speaker, phrase)
        self.suggestion_history = []  # list of (timestamp, speaker, suggestion)
        self._stt_generation = 0
        self._history_lock = threading.Lock()

        # Load Persisted Config
        self.load_config()

        # Initialize Engines
        self.init_chromadb()
        self.init_llm_client()
        self.init_whisper_engine()

        # Initial Scan
        self.refresh_profiles()
        self.refresh_audio_devices()
        self.refresh_review_packets()

        # Reactive Subscriptions
        self.selected_profile.subscribe(self.on_profile_selected)
        self.llm_provider.subscribe(self.on_provider_changed)
        self.selected_model.subscribe(self.on_model_changed)
        self.stt_language.subscribe(self.on_language_changed)
        self.whisper_model_option.subscribe(self.on_whisper_model_changed)
        self.selected_mic.subscribe(lambda _value: self.on_audio_device_changed())
        self.selected_meeting.subscribe(lambda _value: self.on_audio_device_changed())

    # --- CONFIGURATION PERSISTENCE & DEFAULTS ---
    def load_config(self):
        """Loads configuration from config.json, falling back cleanly to DEFAULT_CONFIG."""
        cfg = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"[Config Load Error]: {e}", file=sys.stderr)

        env_key = os.getenv("OPENAI_API_KEY", "")
        default_provider = "openai" if env_key else DEFAULT_CONFIG["provider"]
        default_model = "gpt-4o-mini" if env_key else DEFAULT_CONFIG["selected_model"]

        self.llm_provider.set(cfg.get("provider", default_provider))
        self.ollama_url_input.set(cfg.get("ollama_url", DEFAULT_CONFIG["ollama_url"]))
        self.openai_key_input.set(cfg.get("openai_api_key", env_key or DEFAULT_CONFIG["openai_api_key"]))
        self.custom_url_input.set(cfg.get("custom_url", DEFAULT_CONFIG["custom_url"]))
        self.custom_key_input.set(cfg.get("custom_api_key", DEFAULT_CONFIG["custom_api_key"]))
        self.selected_model.set(cfg.get("selected_model", default_model))
        self.llm_model_name = self.selected_model()

        loaded_prompt = cfg.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
        if not loaded_prompt or ("CRITICAL OPERATIONAL RULES:" in loaded_prompt and "DIRECT MEMORY ANCHORING" not in loaded_prompt):
            loaded_prompt = DEFAULT_SYSTEM_PROMPT
        self.system_prompt.set(loaded_prompt)
        loaded_temp = cfg.get("temperature", DEFAULT_CONFIG["temperature"])
        if loaded_temp == 0.7:
            loaded_temp = 0.0
        self.temperature.set(str(loaded_temp))
        self.max_tokens.set(str(cfg.get("max_tokens", DEFAULT_CONFIG["max_tokens"])))
        self.silence_threshold.set(str(cfg.get("silence_threshold", DEFAULT_CONFIG["silence_threshold"])))
        self.stt_language.set(cfg.get("stt_language", DEFAULT_CONFIG["stt_language"]))
        is_en = (self.stt_language() == "en")
        choices = ENGLISH_MODELS if is_en else MULTILINGUAL_MODELS
        self.whisper_model_choices.set(choices)
        self.whisper_model_option.set(cfg.get("whisper_model", DEFAULT_CONFIG["whisper_model"]))
        mic_choice, meeting_choice = self._audio_choices_from_config(cfg)
        self.selected_mic.set(mic_choice)
        self.selected_meeting.set(meeting_choice)

    def save_config(self):
        """Saves current provider, URLs, credentials, model, system prompt, and parameters to config.json."""
        try:
            temp_val = float(str(self.temperature()).strip())
        except ValueError:
            temp_val = DEFAULT_CONFIG["temperature"]

        try:
            tokens_val = int(str(self.max_tokens()).strip())
        except ValueError:
            tokens_val = DEFAULT_CONFIG["max_tokens"]

        try:
            vad_val = float(str(self.silence_threshold()).strip())
        except ValueError:
            vad_val = DEFAULT_CONFIG["silence_threshold"]

        cfg = {
            "provider": self.llm_provider(),
            "ollama_url": self.ollama_url_input().strip(),
            "openai_api_key": self.openai_key_input().strip(),
            "custom_url": self.custom_url_input().strip(),
            "custom_api_key": self.custom_key_input().strip(),
            "selected_model": self.selected_model().strip(),
            "stt_language": self.stt_language().strip(),
            "whisper_model": self.whisper_model_option().strip(),
            "system_prompt": self.system_prompt().strip(),
            "temperature": temp_val,
            "max_tokens": tokens_val,
            "silence_threshold": vad_val,
            "mic_device": self.selected_mic() or "auto",
            "meeting_device": self.selected_meeting() or "auto",
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            print(f"[Config Save Error]: {e}", file=sys.stderr)

    def save_prompt_and_params(self):
        """Explicitly saves custom system prompt and generation parameters to config."""
        self.save_config()
        self.notify("System prompt and parameters saved successfully!", tone="$success")

    def reset_system_prompt_to_default(self):
        """Restores system prompt to the standard application default."""
        self.system_prompt.set(DEFAULT_SYSTEM_PROMPT)
        self.save_config()
        self.notify("System prompt restored to standard default.", tone="$primary")

    def reset_all_to_defaults(self):
        """Restores all configurable settings back to application defaults."""
        self.llm_provider.set(DEFAULT_CONFIG["provider"])
        self.ollama_url_input.set(DEFAULT_CONFIG["ollama_url"])
        self.openai_key_input.set(DEFAULT_CONFIG["openai_api_key"])
        self.custom_url_input.set(DEFAULT_CONFIG["custom_url"])
        self.custom_key_input.set(DEFAULT_CONFIG["custom_api_key"])
        self.selected_model.set(DEFAULT_CONFIG["selected_model"])
        self.stt_language.set(DEFAULT_CONFIG["stt_language"])
        self.whisper_model_choices.set(ENGLISH_MODELS)
        self.whisper_model_option.set(DEFAULT_CONFIG["whisper_model"])
        self.system_prompt.set(DEFAULT_CONFIG["system_prompt"])
        self.temperature.set(str(DEFAULT_CONFIG["temperature"]))
        self.max_tokens.set(str(DEFAULT_CONFIG["max_tokens"]))
        self.silence_threshold.set(str(DEFAULT_CONFIG["silence_threshold"]))
        self.selected_mic.set(DEFAULT_CONFIG["mic_device"])
        self.selected_meeting.set(DEFAULT_CONFIG["meeting_device"])
        self.save_config()
        self.fetch_available_models(silent=True)
        self.init_whisper_engine()
        self.notify("All settings restored to application defaults.", tone="$success")

    # --- THEME MANAGEMENT ---
    def toggle_theme(self):
        """Toggles between Pastel Dark and Pastel Light themes."""
        self.current_theme.set(PASTEL_LIGHT if self.current_theme() is PASTEL_DARK else PASTEL_DARK)

    # --- TOAST NOTIFICATIONS ---
    def notify(self, message: str, tone: str = "$accent", duration_sec: float = 3.5):
        """Displays floating toast notification that auto-dismisses."""
        safe_tone = tone if tone.startswith("$") else f"${tone}"
        self.toast_tone.set(safe_tone)
        self.toast_msg.set(message)

        def dismiss():
            time.sleep(duration_sec)
            if self.toast_msg() == message:
                self.toast_msg.set(None)

        threading.Thread(target=dismiss, daemon=True).start()

    # --- ENGINE INITIALIZATION ---
    def init_chromadb(self):
        """Initializes persistent ChromaDB client."""
        try:
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        except Exception as e:
            print(f"[ChromaDB Init Error]: {e}", file=sys.stderr)
            self.notify(f"ChromaDB Error: {e}", tone="$danger")

    def init_llm_client(self):
        """Initializes OpenAI client targeting Ollama, official OpenAI, or custom OpenAI-compatible server."""
        self.fetch_available_models(silent=True)

    def on_provider_changed(self, provider_name: str):
        """Handles user switching provider dropdown."""
        self.save_config()
        self.fetch_available_models(silent=False)

    def on_model_changed(self, model_name: str):
        """Handles user picking a different model from dropdown."""
        if not model_name:
            return
        self.llm_model_name = model_name
        p_name = self.llm_provider().capitalize()
        self.engine_model_label.set(f"{p_name} ({model_name})")
        self.save_config()

    def fetch_available_models(self, silent: bool = False):
        """Queries remote model list endpoint (/api/tags or /v1/models) and updates dropdown options."""
        def worker():
            provider = self.llm_provider()
            models_found = []
            err_msg = None

            if not silent:
                self.notify("Fetching remote models list...", tone="$primary", duration_sec=4.0)

            # Determine endpoint and credentials
            if provider == "ollama":
                raw_url = self.ollama_url_input().strip().rstrip("/")
                if not raw_url:
                    raw_url = "http://localhost:11434"
                    self.ollama_url_input.set(raw_url)

                # Normalize host and SDK URL
                host = raw_url[:-3] if raw_url.endswith("/v1") else raw_url
                sdk_url = f"{host}/v1"

                # 1. Try Ollama native /api/tags
                try:
                    req = urllib.request.urlopen(f"{host}/api/tags", timeout=2.5)
                    data = json.loads(req.read().decode())
                    tags = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    if tags:
                        models_found = tags
                except Exception as e:
                    err_msg = str(e)

                # 2. Try OpenAI-compatible endpoint
                if not models_found:
                    try:
                        temp_client = OpenAI(base_url=sdk_url, api_key="ollama")
                        models_found = [m.id for m in temp_client.models.list().data]
                    except Exception as e:
                        err_msg = str(e)

                # Set client
                self.llm_client = OpenAI(base_url=sdk_url, api_key="ollama")

            elif provider == "openai":
                api_key = self.openai_key_input().strip() or os.getenv("OPENAI_API_KEY", "")
                if not api_key:
                    err_msg = "Please provide an OpenAI API Key."
                else:
                    try:
                        temp_client = OpenAI(api_key=api_key)
                        raw_models = [m.id for m in temp_client.models.list().data]
                        priority = ["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
                        chat_models = [m for m in priority if m in raw_models]
                        other_models = [m for m in raw_models if m not in chat_models and "gpt" in m.lower()]
                        models_found = (chat_models + other_models) if chat_models else raw_models
                        self.llm_client = temp_client
                    except Exception as e:
                        err_msg = str(e)

            elif provider == "custom":
                raw_url = self.custom_url_input().strip().rstrip("/")
                if not raw_url:
                    raw_url = "http://localhost:1234/v1"
                    self.custom_url_input.set(raw_url)

                api_key = self.custom_key_input().strip() or "none"
                try:
                    temp_client = OpenAI(base_url=raw_url, api_key=api_key)
                    models_found = [m.id for m in temp_client.models.list().data]
                    self.llm_client = temp_client
                except Exception as e:
                    err_msg = str(e)

            # Process discovered models
            if models_found:
                # Update dropdown options
                opts = [(m, m) for m in models_found]
                self.available_models.set(opts)

                # Verify current selection
                current_sel = self.selected_model()
                if not current_sel or current_sel not in models_found:
                    # Prefer llama or gpt if available, else first
                    best = next((m for m in models_found if "llama3.1" in m or "gpt-4o" in m), models_found[0])
                    self.selected_model.set(best)
                    self.llm_model_name = best
                else:
                    self.llm_model_name = current_sel

                p_label = provider.capitalize()
                self.engine_model_label.set(f"{p_label} ({self.llm_model_name})")
                self.model_status_note.set(f"Connected to {p_label}. Discovered {len(models_found)} models.")
                self.save_config()

                if not silent:
                    self.notify(f"Discovered {len(models_found)} models from {p_label}!", tone="$success")
            else:
                self.model_status_note.set(f"Connection failed: {err_msg or 'No models returned'}")
                if not silent:
                    self.notify(f"Model query failed: {err_msg}", tone="$danger", duration_sec=5.0)

        threading.Thread(target=worker, daemon=True).start()

    def apply_model_config(self):
        """Applies configuration and re-queries models."""
        self.save_config()
        self.fetch_available_models(silent=False)

    def init_whisper_engine(self):
        """Initializes Faster-Whisper fast & accurate model in background."""
        def load():
            with self._whisper_lock:
                model_tag = self.whisper_model_option() or "small.en"
                if self.whisper_model is not None and getattr(self, "_loaded_whisper_tag", None) == model_tag:
                    return

                try:
                    self.status_text.set(f"LOADING STT ({model_tag})")
                    self.status_color.set("#7DD3FC")  # Pastel sky blue
                    print(f"[STT Engine] Loading Faster-Whisper {model_tag} with optimized speed settings...")
                    threads = max(4, min(8, os.cpu_count() or 4))
                    self.whisper_model = WhisperModel(
                        model_tag,
                        device="cpu",
                        compute_type="int8_float32",
                        cpu_threads=threads,
                        num_workers=1
                    )
                    self._loaded_whisper_tag = model_tag
                    print(f"[STT Engine] Faster-Whisper {model_tag} loaded successfully.")
                    self.status_text.set("SYSTEM READY")
                    self.status_color.set("#86EFAC")
                except Exception as e:
                    print(f"[STT Error] Whisper {model_tag} loading failed: {e}", file=sys.stderr)
                    try:
                        fallback_tag = "base.en" if self.stt_language() == "en" else "base"
                        print(f"[STT Engine] Falling back to standard int8 {fallback_tag}...")
                        self.whisper_model = WhisperModel(fallback_tag, device="cpu", compute_type="int8")
                        self._loaded_whisper_tag = fallback_tag
                        self.status_text.set(f"SYSTEM READY ({fallback_tag.upper()})")
                        self.status_color.set("#86EFAC")
                    except Exception as ex2:
                        self.status_text.set("STT FAILED")
                        self.status_color.set("#FDA4AF")
                        self.notify(f"Whisper STT Init Error: {ex2}", tone="$danger")

        threading.Thread(target=load, daemon=True).start()

    def on_language_changed(self, new_lang: str):
        """Switches available models between English-optimized and Multilingual."""
        is_en = (new_lang == "en")
        new_choices = ENGLISH_MODELS if is_en else MULTILINGUAL_MODELS
        self.whisper_model_choices.set(new_choices)

        current_model = self.whisper_model_option()
        available_tags = [tag for tag, _ in new_choices]

        if current_model not in available_tags:
            new_model = "small.en" if is_en else "small"
            self.whisper_model_option.set(new_model)
        else:
            self.init_whisper_engine()
        self.save_config()

    def on_whisper_model_changed(self, new_model_val: str):
        """Re-initializes speech-to-text engine when a new model is selected."""
        self.save_config()
        self.init_whisper_engine()

    # --- PROFILE MANAGEMENT ---
    def refresh_profiles(self):
        """Scans profile directory and updates selector options."""
        subdirs = [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d)) and not d.startswith('.')]
        if not subdirs:
            default_path = os.path.join(PROFILES_DIR, "Default_Client")
            os.makedirs(default_path, exist_ok=True)
            subdirs = ["Default_Client"]

        subdirs.sort()
        opts = [(name, name) for name in subdirs]
        self.profiles_options.set(opts)

        current = self.selected_profile()
        if not current or current not in subdirs:
            self.selected_profile.set(subdirs[0])
        else:
            self.on_profile_selected(current)

    def on_profile_selected(self, active_name: str):
        """Switches active vector database collection and loads chunk previews."""
        if not active_name or self.chroma_client is None:
            return
        safe_name = "prof_" + re.sub(r'[^a-zA-Z0-9_-]', '_', active_name).lower()[:50]
        try:
            self.collection = self.chroma_client.get_or_create_collection(name=safe_name)
            count = self.collection.count()
            self.doc_count.set(count)

            try:
                docs = self.collection.get(limit=50).get("documents", [])
                self.profile_chunks.set(docs)
            except Exception:
                self.profile_chunks.set([])

            self.suggestion_text.set(
                f"Active Profile: [{active_name}]\n"
                f"Knowledge Memory: {count} indexed context chunks.\n\n"
                f"Start call session to view real-time grounded suggestions."
            )
            self.bottom_info.set(f"Profile: {active_name} | Indexed Fragments: {count} | Grounding Active.")
        except Exception as e:
            print(f"[ChromaDB Switch Error]: {e}", file=sys.stderr)
            self.notify(f"Failed to open collection for {active_name}: {e}", tone="$danger")

    def create_new_profile(self):
        """Creates a new client profile folder and switches to it."""
        raw_name = self.new_profile_name().strip()
        cleaned = re.sub(r'\s+', '_', raw_name)
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', cleaned)
        if not cleaned:
            self.notify("Please enter a valid profile name.", tone="$danger")
            return

        target_dir = os.path.join(PROFILES_DIR, cleaned)
        os.makedirs(target_dir, exist_ok=True)
        self.new_profile_name.set("")
        self.show_new_profile_dialog.set(False)
        self.refresh_profiles()
        self.selected_profile.set(cleaned)
        self.notify(f"Profile [{cleaned}] created successfully!", tone="$success")

    def reset_profile_memory(self):
        """Erases all indexed vectors for the active profile."""
        active = self.selected_profile()
        if not active or self.chroma_client is None:
            return
        safe_name = "prof_" + re.sub(r'[^a-zA-Z0-9_-]', '_', active).lower()[:50]
        try:
            self.chroma_client.delete_collection(name=safe_name)
            self.collection = self.chroma_client.get_or_create_collection(name=safe_name)
            self.doc_count.set(0)
            self.profile_chunks.set([])
            self.retrieved_sources.set([])
            self.suggestion_text.set(f"Context wiped for profile [{active}]. Total entries: 0.")
            self.bottom_info.set(f"Profile: {active} | Indexed Fragments: 0 | Memory Reset.")
            self.show_reset_dialog.set(False)
            self.notify(f"Memory wiped for [{active}].", tone="$success")
        except Exception as e:
            self.notify(f"Reset failed: {e}", tone="$danger")

    # --- CONTEXT INGESTION & CHATGPT PARSING ---
    def import_context_file(self):
        """Opens native file picker to ingest ChatGPT conversation export."""
        path = dialogs.open_file(
            title="Select ChatGPT Export File",
            filters=[
                ("ChatGPT Exports (*.json, *.zip, *.md, *.txt)", ["json", "zip", "md", "txt"]),
                ("JSON Files (*.json)", ["json"]),
                ("ZIP Archives (*.zip)", ["zip"]),
                ("Markdown / Text (*.md, *.txt)", ["md", "txt"]),
                ("All Files (*)", ["*"])
            ]
        )
        if not path:
            return

        threading.Thread(target=self._process_and_index_file, args=(path,), daemon=True).start()

    def _process_and_index_file(self, file_path: str):
        """Parses exported files, segments dialogue nodes, and loads into ChromaDB."""
        self.notify("Parsing context export... Please wait.", tone="$primary", duration_sec=8.0)
        extracted_chunks = []
        try:
            filename = os.path.basename(file_path).lower()

            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as archive:
                    if 'conversations.json' in archive.namelist():
                        with archive.open('conversations.json') as f:
                            raw_data = json.load(f)
                            extracted_chunks.extend(self._parse_chatgpt_json_structure(raw_data))

            elif filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    extracted_chunks.extend(self._parse_chatgpt_json_structure(raw_data))

            elif filename.endswith('.md') or filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    extracted_chunks = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 25]

            if not extracted_chunks:
                raise ValueError("No viable conversation messages or paragraphs found in file.")

            active = self.selected_profile()
            timestamp = int(time.time())
            ids = [f"{active}_{timestamp}_{i}" for i in range(len(extracted_chunks))]
            self.collection.add(documents=extracted_chunks, ids=ids)

            count = self.collection.count()
            self.doc_count.set(count)
            try:
                self.profile_chunks.set(self.collection.get(limit=50).get("documents", []))
            except Exception:
                pass

            self.suggestion_text.set(
                f"Profile Context Refreshed: [{active}]\n"
                f"Total Indexed Knowledge Fragments: {count}\n\n"
                f"Assistant is primed and ready to support your live calls."
            )
            self.bottom_info.set(f"Profile: {active} | Indexed Fragments: {count} | Ingestion Complete.")
            self.notify(f"Imported {len(extracted_chunks)} fragments into [{active}]!", tone="$success")

        except Exception as e:
            print(f"[Context Ingestion Error]: {e}", file=sys.stderr)
            self.notify(f"Import failed: {e}", tone="$danger")

    def _parse_chatgpt_json_structure(self, json_data) -> List[str]:
        """Parses OpenAI's conversation tree into clean grounding text entries."""
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
                    t = " ".join([p for p in parts if isinstance(p, str)]).strip()
                    if t and len(t) > 15:
                        entry = f"[Thread: {title}] ({role}): {t}"
                        parsed.append(entry)

        return parsed

    # --- AUDIO DEVICE ENUMERATION & SELECTION ---
    def refresh_audio_devices(self):
        """Builds separate microphone and meeting-audio lists."""
        try:
            devices = sd.query_devices()
        except Exception as e:
            print(f"[Audio Device Query Error]: {e}", file=sys.stderr)
            return

        physical = []
        loopbacks = []
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] <= 0:
                continue
            label = short_device_label(dev['name'])
            bucket = loopbacks if self._is_loopback_device(dev['name']) else physical
            bucket.append((str(idx), label))

        multi_outputs = [(device["id"], device["label"]) for device in multi_output_devices()]
        mic_opts = [("auto", "System default microphone")] + physical + loopbacks
        meeting_opts = [("auto", "BlackHole if available")] + multi_outputs + loopbacks + physical
        self.mic_options.set(mic_opts)
        self.meeting_options.set(meeting_opts)
        mic_ids = {val for val, _ in mic_opts}
        meeting_ids = {val for val, _ in meeting_opts}
        if self.selected_mic() not in mic_ids:
            self.selected_mic.set("auto")
        if self.selected_meeting() not in meeting_ids:
            self.selected_meeting.set("auto")

    def on_audio_device_changed(self):
        """Persists the two audio source choices."""
        self._apply_audio_selection_label()
        self.save_config()

    def _audio_choices_from_config(self, cfg):
        """Reads the two saved sources. Older configs stored a single device."""
        mic = str(cfg.get("mic_device", DEFAULT_CONFIG["mic_device"]))
        meeting = str(cfg.get("meeting_device", DEFAULT_CONFIG["meeting_device"]))
        if "mic_device" not in cfg and "meeting_device" not in cfg:
            legacy = str(cfg.get("selected_audio", "auto"))
            if legacy and legacy != "auto":
                try:
                    dev = sd.query_devices(int(legacy))
                    if self._is_loopback_device(dev['name']):
                        meeting = legacy
                    else:
                        mic = legacy
                except Exception:
                    pass
        return mic or "auto", meeting or "auto"

    def _lookup_input_device(self, choice: str):
        if not choice or choice == "auto":
            return None
        try:
            idx = int(choice)
            dev = sd.query_devices(idx)
        except Exception:
            return None
        if dev['max_input_channels'] <= 0:
            return None
        return idx, dev

    def _apply_audio_selection_label(self):
        try:
            sources = self.resolve_capture_sources()
        except Exception:
            self.active_device_label.set("No input devices")
            return
        self.active_device_label.set(" + ".join(
            "Mic" if s["speaker"] == "You" else "Meeting" for s in sources
        ))

    @staticmethod
    def _is_loopback_device(name: str) -> bool:
        return "blackhole" in (name or "").lower()

    def resolve_capture_sources(self):
        """Opens the microphone and meeting device chosen in Audio Hardware."""
        devices = list(sd.query_devices())
        inputs = [(idx, dev) for idx, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        if not inputs:
            raise RuntimeError("No input audio devices found.")

        microphones = [(idx, dev) for idx, dev in inputs if not self._is_loopback_device(dev['name'])]

        mic = self._lookup_input_device(self.selected_mic())
        loop = resolve_meeting_device(self.selected_meeting())

        if mic is None:
            default_in = sd.default.device[0]
            if default_in is not None and int(default_in) >= 0:
                try:
                    default_dev = sd.query_devices(int(default_in))
                    if default_dev['max_input_channels'] > 0 and not self._is_loopback_device(default_dev['name']):
                        mic = (int(default_in), default_dev)
                except Exception:
                    pass
            if mic is None and microphones:
                mic = microphones[0]

        sources = []
        if mic is not None:
            sources.append({"index": mic[0], "info": mic[1], "speaker": "You"})
        if loop is not None and (mic is None or loop[0] != mic[0]):
            sources.append({"index": loop[0], "info": loop[1], "speaker": "Participant"})
        if not sources:
            idx, dev = inputs[0]
            sources.append({
                "index": idx,
                "info": dev,
                "speaker": "Participant" if self._is_loopback_device(dev['name']) else "You",
            })
        return sources

    def _close_audio_streams(self):
        streams = list(getattr(self, "streams", None) or [])
        if self.stream is not None and self.stream not in streams:
            streams.append(self.stream)
        for stream in streams:
            try:
                if stream.active:
                    stream.stop()
                stream.close()
            except Exception:
                pass
        self.streams = []
        self.stream = None

    def _make_audio_callback(self, speaker: str, samplerate: int):
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[Audio Status] {speaker}: {status}", file=sys.stderr)
            self.audio_queue.put((speaker, samplerate, to_mono(indata)))
        return callback

    # --- LIVE AUDIO & TRANSCRIPTION PIPELINE ---
    def audio_callback(self, indata, frames, time_info, status):
        """Single-stream fallback. Live sessions use per-source callbacks."""
        if status:
            print(f"[Audio Status] {status}", file=sys.stderr)
        self.audio_queue.put(("Participant", self.current_samplerate or 16000, to_mono(indata)))

    def start_session(self):
        """Opens the microphone and BlackHole together, then starts the STT loop."""
        if self.is_listening():
            return

        if self.whisper_model is None:
            self.notify("Whisper STT is still loading. Please wait a moment...", tone="$primary")
            return

        if self.doc_count() == 0:
            self.notify(f"Profile [{self.selected_profile()}] has 0 imported documents.", tone="$primary")

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        opened = []
        try:
            sources = self.resolve_capture_sources()
            errors = []
            for source in sources:
                dev_info = source["info"]
                samplerate = int(dev_info.get('default_samplerate', 16000))
                channels = 2 if dev_info['max_input_channels'] >= 2 and source["speaker"] == "Participant" else 1
                channels = min(channels, int(dev_info['max_input_channels']))
                blocksize = int(samplerate * 0.25)
                try:
                    stream = sd.InputStream(
                        device=source["index"],
                        channels=channels,
                        samplerate=samplerate,
                        callback=self._make_audio_callback(source["speaker"], samplerate),
                        blocksize=blocksize,
                    )
                    stream.start()
                    opened.append(stream)
                    source["samplerate"] = samplerate
                    print(f"[Audio Stream] {source['speaker']} on [{source['index']}] {dev_info['name']} ({samplerate}Hz, {channels}ch)")
                except Exception as e:
                    errors.append(f"{dev_info['name']}: {e}")
                    print(f"[Audio Stream] Could not open {dev_info['name']}: {e}", file=sys.stderr)

            if not opened:
                detail = "; ".join(errors) or "No input stream could be started."
                raise RuntimeError(detail)

            self.streams = opened
            self.stream = opened[0]
            live_sources = [s for s in sources if "samplerate" in s]
            self._has_meeting_stream = any(s["speaker"] == "Participant" for s in live_sources)
            self.current_samplerate = live_sources[0]["samplerate"]
            names = [f"{s['speaker']}: {s['info']['name']}" for s in live_sources]
            rates = [f"{s['speaker']} {s['samplerate']}Hz" for s in live_sources]
            self.active_device_label.set(" + ".join(
                "BlackHole" if self._is_loopback_device(s["info"]["name"]) else "Mic" for s in live_sources
            ))
            self.hardware_sr_label.set(" · ".join(rates) + " (-> 16kHz)")

            self.is_listening.set(True)
            self.status_text.set("LIVE CALL ACTIVE")
            self.status_color.set("#FDA4AF")
            self.bottom_info.set("Monitoring: " + " | ".join(names) + " | Zero-hallucination suggestions active.")
            with self._history_lock:
                self._stt_generation += 1
                self.transcript_history.clear()
                self.suggestion_history.clear()

            self._start_stt_worker()
            self.processing_thread = threading.Thread(target=self._process_audio_pipeline, daemon=True)
            self.processing_thread.start()
            if errors:
                self.notify("One source could not be opened: " + "; ".join(errors), tone="$primary")
            else:
                self.notify("Listening to your mic and the meeting.", tone="$success")

        except Exception as e:
            self._close_audio_streams()
            print(f"[Audio Stream Start Error]: {e}", file=sys.stderr)
            self.notify(f"Could not open audio devices: {e}", tone="$danger")
            if self.is_listening():
                self.stop_session()

    def stop_session(self):
        """Stops both audio streams, exports the review packet, and returns to idle."""
        if not self.is_listening():
            return

        self.is_listening.set(False)
        self._close_audio_streams()
        vad_thread = getattr(self, "processing_thread", None)
        if vad_thread is not None and vad_thread.is_alive():
            vad_thread.join(timeout=1.0)
        self._stop_stt_worker()

        self.status_text.set("SESSION STOPPED")
        self.status_color.set("#94A3B8")
        self.bottom_info.set("Session concluded. Finishing any transcripts still in progress.")

    def _vad_take_phrase(self, state, chunk):
        """Adaptive trailing-window VAD for one source. Returns a phrase or None."""
        return take_speech_phrase(state, chunk)

    def _start_stt_worker(self):
        """One background worker. Capture never waits on it."""
        previous = getattr(self, "_stt_queue", None)
        if previous is not None:
            previous.put_nowait(None)
        self._stt_queue = queue.Queue()
        self._stt_thread = threading.Thread(
            target=self._stt_loop, args=(self._stt_queue,), daemon=True, name="stt"
        )
        self._stt_thread.start()

    def _stt_loop(self, pending: queue.Queue):
        while True:
            item = pending.get()
            if item is None:
                return
            generation, phrase, sr, speaker = item
            if generation != self._stt_generation:
                continue
            try:
                self._process_phrase(phrase, sr, speaker, generation)
            except Exception as exc:
                print(f"[Transcription Error]: {exc}", file=sys.stderr)

    def _submit_phrase(self, phrase, sr: int, speaker: str):
        """Queue a copy of the phrase. This returns immediately."""
        pending = getattr(self, "_stt_queue", None)
        if pending is None:
            return
        pending.put_nowait((
            self._stt_generation,
            np.ascontiguousarray(phrase, dtype=np.float32),
            int(sr),
            speaker,
        ))

    def _stop_stt_worker(self):
        """Finish queued phrases in the background, then save, without freezing the window."""
        pending = getattr(self, "_stt_queue", None)
        thread = getattr(self, "_stt_thread", None)
        generation = self._stt_generation
        if pending is not None:
            pending.put_nowait(None)

        def finish():
            if thread is not None and thread.is_alive():
                thread.join(timeout=60)
            with self._history_lock:
                if self._stt_generation != generation:
                    return
            self.export_post_call_packet()
            self.refresh_review_packets()
            self.bottom_info.set("Session concluded. Review packet exported to ~/.context_assistant/review_packets.")

        threading.Thread(target=finish, daemon=True).start()

    def _is_echo(self, speaker: str, text: str) -> bool:
        previous = self.transcript_history[-1] if self.transcript_history else None
        if previous is None:
            return False
        _stamp, previous_speaker, previous_text = previous
        return is_cross_talk_echo(previous_speaker, previous_text, speaker, text)

    def _process_audio_pipeline(self):
        """Buffers each source separately so your voice and theirs are transcribed apart."""
        try:
            cfg_thresh = float(str(self.silence_threshold()).strip())
        except Exception:
            cfg_thresh = 0.015

        states = {}
        print(f"[Audio Pipeline] Dual-source VAD (baseline={cfg_thresh})")

        while self.is_listening():
            try:
                speaker, sr, chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            state = states.get(speaker)
            if state is None or state["sr"] != sr:
                state = {
                    "buffer": np.zeros(0, dtype=np.float32),
                    "noise_floor": 0.0015,
                    "is_speaking": False,
                    "silent_chunks": 0,
                    "sr": int(sr),
                    "cfg_thresh": cfg_thresh,
                }
                states[speaker] = state

            phrase = self._vad_take_phrase(state, chunk)
            if phrase is not None:
                print(f"[Audio Pipeline] {speaker} phrase ({len(phrase)/sr:.2f}s) -> queued")
                self._submit_phrase(phrase, int(sr), speaker)

    def _process_phrase(self, audio_data: np.ndarray, samplerate=None, speaker: str = "Participant", generation: int = None):
        """Resamples to 16kHz float32, normalizes audio volume, and runs fast, accurate Whisper transcription."""
        if generation is None:
            generation = self._stt_generation
        if generation != self._stt_generation:
            return
        if len(audio_data) < 4000:
            return

        model = self.whisper_model
        if model is None:
            return

        sr = samplerate or self.current_samplerate or 16000
        audio_16k = prepare_speech_audio(audio_data, sr)

        try:
            phrase_text = transcribe_speech(
                model, audio_16k, self.stt_language(), getattr(self, "_loaded_whisper_tag", "")
            )
            if generation != self._stt_generation:
                return
            if phrase_text and self._is_echo(speaker, phrase_text):
                print(f"[Transcription] Dropped echo from {speaker}: {phrase_text}")
                phrase_text = ""
            if phrase_text:
                timestamp = time.strftime("%H:%M:%S")
                with self._history_lock:
                    if generation != self._stt_generation:
                        return
                    self.transcript_history.append((timestamp, speaker, phrase_text))
                    current_list = list(self.transcripts())
                    current_list.insert(0, {"time": timestamp, "speaker": speaker, "text": phrase_text})
                    self.transcripts.set(current_list)
                print(f"[Transcription] [{timestamp}] {speaker}: {phrase_text}")

                if speaker != "You" or not self._has_meeting_stream:
                    print(f"[Suggestion Engine] Querying context for: '{phrase_text}'")
                    threading.Thread(target=self.generate_contextual_suggestion, args=(phrase_text, speaker), daemon=True).start()

        except Exception as e:
            print(f"[Transcription Error]: {e}", file=sys.stderr)

    # --- RAG GROUNDED SUGGESTION ENGINE ---
    def generate_contextual_suggestion(self, incoming_text: str, speaker: str = "Participant"):
        """Queries ChromaDB with multi-result deduplication and prompts LLM with direct memory anchoring."""
        context_str = "No specific history entries discovered for this profile target."
        sources = []

        if self.collection and self.collection.count() > 0:
            try:
                # Build search queries: incoming text and recent turns for reference resolution
                search_queries = [incoming_text]
                try:
                    recent_items = self.transcripts()[:2] if hasattr(self, "transcripts") else []
                    if recent_items:
                        prev_text = recent_items[0].get("text", "").strip()
                        if prev_text and prev_text != incoming_text:
                            search_queries.append(f"{prev_text} {incoming_text}")
                except Exception:
                    pass

                results = self.collection.query(query_texts=search_queries, n_results=6)
                if results.get('documents'):
                    seen_docs = set()
                    for doc_list in results['documents']:
                        for doc in doc_list:
                            clean_doc = doc.strip()
                            if clean_doc and clean_doc not in seen_docs:
                                seen_docs.add(clean_doc)
                                sources.append(clean_doc)
                    if sources:
                        context_str = "\n\n".join(
                            [f"[DOCUMENTED MEMORY {i+1}]:\n{d}" for i, d in enumerate(sources)]
                        )
            except Exception as e:
                print(f"[ChromaDB Query Error]: {e}", file=sys.stderr)

        self.retrieved_sources.set(sources)

        # Construct system prompt from custom template or default
        raw_prompt = self.system_prompt().strip() or DEFAULT_SYSTEM_PROMPT
        if "{context}" in raw_prompt:
            final_system_prompt = raw_prompt.replace("{context}", context_str)
        else:
            final_system_prompt = f"{raw_prompt}\n\n=== VERIFIED CLIENT / WORK HISTORY ===\n{context_str}"

        try:
            temp_val = float(str(self.temperature()).strip())
        except Exception:
            temp_val = 0.0

        try:
            tokens_val = int(str(self.max_tokens()).strip())
        except Exception:
            tokens_val = 160

        # Construct user prompt with recent live call context if available
        prior_turns = []
        try:
            recent_items = self.transcripts()[:2] if hasattr(self, "transcripts") else []
            prior_turns = [
                f"- {item.get('speaker', 'Speaker')}: \"{item.get('text', '').strip()}\""
                for item in reversed(recent_items)
                if item.get("text", "").strip() and item.get("text", "").strip() != incoming_text.strip()
            ]
        except Exception:
            pass

        said = "I said" if speaker == "You" else "Participant said"
        if prior_turns:
            user_msg = (
                f"Recent live call context:\n" + "\n".join(prior_turns) +
                f"\n\n{said}: '{incoming_text}'"
            )
        else:
            user_msg = f"{said}: '{incoming_text}'"

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=temp_val,
                max_tokens=tokens_val
            )
            raw_suggestion = response.choices[0].message.content
            suggestion = raw_suggestion.strip() if raw_suggestion else ""
            if suggestion.startswith('"') and suggestion.endswith('"') and len(suggestion) > 2:
                suggestion = suggestion[1:-1].strip()
            if not suggestion:
                suggestion = "I don't have that documented in our prior work history, but let me check and follow up."
            self.suggestion_text.set(suggestion)
            self.suggestion_history.append((time.strftime("%H:%M:%S"), speaker, suggestion))
            print(f"[Suggestion Generated]: {suggestion}")
        except Exception as e:
            err_msg = (
                f"LLM Connection Error:\n"
                f"Ensure your provider server ({self.llm_provider()}) is online.\n\nDetails: {e}"
            )
            self.suggestion_text.set(err_msg)
            print(f"[Suggestion Error]: {e}", file=sys.stderr)

    # --- CLIPBOARD & POST-CALL EXPORT ---
    def copy_suggestion_to_clipboard(self):
        """Copies suggestion text to system clipboard via macOS pbcopy."""
        text_to_copy = self.suggestion_text().strip()
        if not text_to_copy:
            return

        try:
            subprocess.run(["pbcopy"], input=text_to_copy.encode("utf-8"), check=True)
            self.copy_btn_state.set(("Copied", "check"))
            self.notify("Suggestion copied to clipboard!", tone="$success", duration_sec=2.0)

            def revert():
                time.sleep(1.8)
                self.copy_btn_state.set(("Copy", "copy"))

            threading.Thread(target=revert, daemon=True).start()
        except Exception as e:
            self.notify(f"Copy failed: {e}", tone="$danger")

    def open_export_dialog(self, source: str = "live"):
        """Ask whether to save the full conversation or the meeting resume, and in which format."""
        turns, _suggestions, _profile = self._export_payload(source)
        if not turns:
            self.notify("There is no conversation to export yet.", tone="$muted")
            return
        self.export_source.set(source)
        self.show_export_dialog.set(True)

    def _export_payload(self, source: str):
        profile = self.selected_profile()
        if source == "packet":
            content = self.selected_packet_content()
            return turns_from_packet(content), [], profile_from_packet(content, profile)
        turns = list(self.transcript_history)
        if not turns:
            turns = [
                (item.get("time", ""), item.get("speaker", ""), item.get("text", ""))
                for item in reversed(self.transcripts())
                if item.get("text")
            ]
        return turns, list(self.suggestion_history), profile

    def save_call_export(self):
        kind = self.export_kind()
        fmt = self.export_format()
        source = self.export_source()
        self.show_export_dialog.set(False)
        turns, suggestions, profile = self._export_payload(source)
        if not turns:
            self.notify("There is no conversation to export yet.", tone="$muted")
            return
        exports_dir = os.path.join(BASE_DIR, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = next(item[2] for item in FORMATS if item[0] == fmt)
        path = dialogs.save_file(
            title="Export call",
            directory=exports_dir,
            default_name=default_filename(kind, fmt, stamp),
            filters=[
                ("Markdown (*.md)", ["md"]),
                ("Text (*.txt)", ["txt"]),
                ("CSV (*.csv)", ["csv"]),
            ],
        )
        if not path:
            return
        path = ensure_extension(path, fmt)
        chosen = {".md": "md", ".txt": "txt", ".csv": "csv"}.get(os.path.splitext(path)[1].lower(), fmt)
        body = render_document(
            kind,
            chosen,
            turns,
            suggestions if kind == KIND_RESUME else [],
            profile,
            time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
            self.notify(f"Saved {os.path.basename(path)}", tone="$success", duration_sec=5.0)
        except Exception as exc:
            self.notify(f"Export failed: {exc}", tone="$danger")

    def export_post_call_packet(self):
        """Exports structured Markdown packet for direct ChatGPT memory sync."""
        if not self.transcript_history:
            self.notify("Session ended with no spoken dialogue to export.", tone="$muted")
            return

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(REVIEWS_DIR, f"review_packet_{timestamp_str}.md")
        active = self.selected_profile()

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
            self.notify(f"Saved review packet: {os.path.basename(export_path)}", tone="$success", duration_sec=5.0)

        except Exception as e:
            print(f"[Export Error]: {e}", file=sys.stderr)
            self.notify(f"Export failed: {e}", tone="$danger")

    def refresh_review_packets(self):
        """Scans saved review packet files."""
        if not os.path.exists(REVIEWS_DIR):
            return
        files = [f for f in os.listdir(REVIEWS_DIR) if f.endswith(".md")]
        files.sort(reverse=True)
        packet_entries = []
        for fn in files:
            full = os.path.join(REVIEWS_DIR, fn)
            stat_res = os.stat(full)
            dt_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat_res.st_mtime))
            packet_entries.append({"name": fn, "path": full, "time": dt_str, "size": f"{stat_res.st_size} bytes"})

        self.review_packets_list.set(packet_entries)
        if packet_entries and not self.selected_packet_name():
            self.select_packet(packet_entries[0]["path"], packet_entries[0]["name"])

    def select_packet(self, path: str, name: str):
        """Loads chosen review packet content for viewing and copying."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.selected_packet_name.set(name)
            self.selected_packet_content.set(content)
        except Exception as e:
            self.notify(f"Could not read packet: {e}", tone="$danger")

    def copy_packet_sync_prompt(self):
        """Extracts and copies UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT block."""
        content = self.selected_packet_content()
        match = re.search(r'```text\n(UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT[\s\S]*?)```', content)
        text_to_copy = match.group(1).strip() if match else content

        try:
            subprocess.run(["pbcopy"], input=text_to_copy.encode("utf-8"), check=True)
            self.packet_copy_state.set(("Copied Ingestion Block!", "check"))
            self.notify("ChatGPT Ingestion Prompt copied to clipboard!", tone="$success", duration_sec=2.5)

            def revert():
                time.sleep(2.0)
                self.packet_copy_state.set(("Copy ChatGPT Sync Prompt", "copy"))

            threading.Thread(target=revert, daemon=True).start()
        except Exception as e:
            self.notify(f"Copy failed: {e}", tone="$danger")

    def clear_transcripts(self):
        """Clears live dialogue transcript stream."""
        self.transcripts.set([])
        self.transcript_history.clear()
        self.suggestion_history.clear()
        self.retrieved_sources.set([])
        self.suggestion_text.set("Transcript cleared. Waiting for spoken dialogue...")
        self.notify("Session transcript cleared.", tone="$muted")

    def cleanup(self):
        """Gracefully closes stream upon window exit."""
        self.is_listening.set(False)
        self._close_audio_streams()

    def request_close_application(self):
        """Initiates an asynchronous graceful shutdown workflow with UI spinner & status."""
        if self.is_closing():
            return  # Already shutting down

        self.is_closing.set(True)

        def shutdown_worker():
            print("[Shutdown] Initiating graceful application shutdown...")

            # 1. Finalize live call audio session if active
            if self.is_listening():
                self.closing_status.set("Finalizing call session & saving review packet...")
                time.sleep(0.3)
                try:
                    self.stop_session()
                except Exception as e:
                    print(f"[Shutdown Error - Stop Session]: {e}", file=sys.stderr)

            # 2. Persist current application settings & options
            self.closing_status.set("Saving application configuration...")
            time.sleep(0.2)
            try:
                self.save_config()
            except Exception as e:
                print(f"[Shutdown Error - Save Config]: {e}", file=sys.stderr)

            # 3. Clean up audio resources and threads
            self.closing_status.set("Releasing audio streams & hardware hooks...")
            time.sleep(0.2)
            try:
                self.cleanup()
            except Exception as e:
                print(f"[Shutdown Error - Audio Cleanup]: {e}", file=sys.stderr)

            # 4. Final shutdown confirmation
            self.closing_status.set("Exiting application...")
            time.sleep(0.2)
            print("[Shutdown] Graceful shutdown tasks complete. Exiting event loop.")

            # Terminate via Aurora app quit or fallback os._exit
            try:
                if self.app and hasattr(self.app, "quit"):
                    self.app.quit()
                else:
                    os._exit(0)
            except Exception:
                os._exit(0)

        threading.Thread(target=shutdown_worker, daemon=True).start()


# Global Controller Instance
ctrl = AuroraAssistantController()


# --- AURORA COMPONENT VIEWS ---

def build_sidebar() -> Element:
    """Builds refined navigation drawer with pastel vector icons and custom brand emblem."""
    nav_items = [
        ("copilot", "mic", "Live Copilot"),
        ("profiles", "brain", "Context & Profiles"),
        ("packets", "file_text", "Review Packets"),
        ("settings", "settings", "Settings & Audio"),
    ]

    active_page = ctrl.active_page()

    nav_rows = []
    for page_key, icon_name, label in nav_items:
        is_active = active_page == page_key

        count_badge = None
        if page_key == "copilot" and ctrl.is_listening():
            count_badge = pill("LIVE", "#FDA4AF", solid=True)
        elif page_key == "profiles":
            count_badge = pill(str(ctrl.doc_count()), "#A5B4FC")
        elif page_key == "packets":
            pk_count = len(ctrl.review_packets_list())
            if pk_count > 0:
                count_badge = pill(str(pk_count), "#6EE7B7")

        nav_rows.append(
            row(
                key=f"nav-{page_key}",
                gap=12,
                align="center",
                padding=(10, 14),
                radius=10,
                bg="$surface_pressed" if is_active else "#00000000",
                bg_hover="$surface_hover",
                on_click=lambda pk=page_key: ctrl.active_page.set(pk),
            )[
                image(get_svg_icon(icon_name, "#A5B4FC" if is_active else "#94A3B8", 16), width=16, height=16),
                text(label, size=13.5, weight=700 if is_active else 500, color="$text" if is_active else "$muted"),
                spacer(),
                count_badge or text(""),
            ]
        )

    # Active Profile Bottom Card
    prof_card = row(
        align="center", gap=10, padding=(10, 12), radius=10,
        bg="$background", border_width=1, border_color="$border"
    )[
        icon_box("user", "#A5B4FC", size=28, icon_size=14),
        column(gap=1, grow=1)[
            text(ctrl.selected_profile(), size=12, weight=700, color="$text", max_lines=1),
            row(gap=5, align="center")[
                box(width=6, height=6, radius="$full", bg="#86EFAC"),
                text(f"{ctrl.doc_count()} memory chunks", size=10, color="$muted"),
            ]
        ]
    ]

    theme_label = "Daylight Theme" if ctrl.current_theme() is PASTEL_DARK else "Midnight Dark"
    theme_icon = "sun" if ctrl.current_theme() is PASTEL_DARK else "moon"

    return column(width=254, padding=(22, 16), gap=6, bg="$surface", border_right_width=1, border_color="$border")[
        # Custom Modern Application Emblem
        row(gap=12, align="center", padding=(0, 6))[
            image(get_logo_svg(), width=38, height=38),
            column(gap=1)[
                text("Live Copilot", size=16, weight=800, color="$text", letter_spacing=-0.3),
                text("Grounded AI Assistant", size=11, color="$muted"),
            ]
        ],
        box(height=16),
        text("NAVIGATION", size=10, weight=750, color="$muted", letter_spacing=1.2, padding=(0, 10)),
        box(height=2),
        nav_rows,
        spacer(),
        action_btn(
            theme_label,
            theme_icon,
            on_click=ctrl.toggle_theme,
            variant="soft",
            key="theme-toggle-btn",
        ),
        box(height=4),
        prof_card,
        box(height=6),
        # Close Application Button (at bottom of left menu)
        row(
            key="btn-close-app",
            align="center",
            gap=8,
            padding=(8, 14),
            radius=10,
            bg="#FDA4AF20" if not ctrl.is_closing() else "#FDA4AF35",
            bg_hover="#FDA4AF30" if not ctrl.is_closing() else "#FDA4AF35",
            border_width=1,
            border_color="#FDA4AF40",
            on_click=ctrl.request_close_application if not ctrl.is_closing() else None,
            opacity=0.75 if ctrl.is_closing() else 1.0,
        )[
            spinner(size=14, tone="#FDA4AF") if ctrl.is_closing() else image(get_svg_icon("power", "#FDA4AF", 14), width=14, height=14),
            text("Closing..." if ctrl.is_closing() else "Close Application", size=13, weight=650, color="#FDA4AF"),
            spacer(),
            pill("Quit", "#FDA4AF") if not ctrl.is_closing() else spinner(size=10, tone="#FDA4AF"),
        ],
    ]


# --- PAGE 1: LIVE COPILOT COMMAND CENTER ---
def view_copilot() -> Element:
    """Builds the main live-call workstation."""

    # Metrics Strip
    metrics = row(gap=14)[
        card(padding=(12, 16), grow=1, bg="$surface", border_width=1, border_color="$border", radius=12, shadow=(0, 2, 8, "$shadow"))[
            row(align="center", gap=10)[
                icon_box("brain", "#A5B4FC", size=32, icon_size=16),
                column(gap=1)[
                    text("ACTIVE PROFILE", size=10, weight=750, color="$muted", letter_spacing=0.8),
                    text(ctrl.selected_profile(), size=13.5, weight=700, color="$text"),
                ],
                spacer(),
                pill(f"{ctrl.doc_count()} chunks", "#A5B4FC"),
            ]
        ],
        card(padding=(12, 16), grow=1, bg="$surface", border_width=1, border_color="$border", radius=12, shadow=(0, 2, 8, "$shadow"))[
            row(align="center", gap=10)[
                icon_box("headphones", "#7DD3FC", size=32, icon_size=16),
                column(gap=1)[
                    text("AUDIO PIPELINE", size=10, weight=750, color="$muted", letter_spacing=0.8),
                    text(ctrl.hardware_sr_label(), size=13.5, weight=700, color="$text"),
                ],
                spacer(),
                pill("1.5s VAD", "#7DD3FC"),
            ]
        ],
        card(padding=(12, 16), grow=1, bg="$surface", border_width=1, border_color="$border", radius=12, shadow=(0, 2, 8, "$shadow"))[
            row(align="center", gap=10)[
                icon_box("cpu", "#86EFAC", size=32, icon_size=16),
                column(gap=1)[
                    text("INFERENCE ENGINE", size=10, weight=750, color="$muted", letter_spacing=0.8),
                    text(ctrl.engine_model_label(), size=13.5, weight=700, color="$text"),
                ],
                spacer(),
                pill("Zero Hallucination", "#86EFAC", dot_color="#86EFAC"),
            ]
        ],
    ]

    # Left Panel: Live Speech Stream
    transcript_items = ctrl.transcripts()
    if not transcript_items:
        left_body = card(padding=36, gap=14, bg="$background", align="center", justify="center", grow=1, radius=12)[
            icon_box("mic", "#A5B4FC", size=52, icon_size=26),
            text("Audio Stream Monitoring", size=15, weight=750, color="$text"),
            text("When live participants speak, conversation phrases are segmented with energy VAD and transcribed here.", size=12, color="$muted", align="center"),
            pill("Auto 16kHz Resampling Active", "#6EE7B7"),
        ]
    else:
        cards_list = []
        for idx, item in enumerate(transcript_items):
            is_latest = (idx == 0)
            cards_list.append(
                card(
                    padding=12, gap=6, bg="$background",
                    border_width=1.5 if is_latest else 1,
                    border_color="#A5B4FC" if is_latest else "$border",
                    radius=10
                )[
                    row(align="center", gap=8)[
                        pill(item["time"], "#94A3B8"),
                        pill(item["speaker"], "#A5B4FC"),
                        pill("LATEST", "#A5B4FC", solid=True) if is_latest else text(""),
                        spacer(),
                    ],
                    text(item["text"], size=13.5, weight=500 if is_latest else 400, color="$text", line_height=1.45)
                ]
            )
        left_body = scroll(key="transcript-scroll-area", grow=1, gap=8)[cards_list]

    left_panel = card(grow=1, width=0, padding=18, gap=12, bg="$surface", border_width=1, border_color="$border", radius=14, shadow=(0, 4, 16, "$shadow"))[
        row(align="center", gap=10)[
            icon_box("mic", "#A5B4FC", size=26, icon_size=13),
            text("Live Call Dialogue Stream", size=15, weight=750, color="$text"),
            badge(f"{len(transcript_items)} phrases", tone="primary"),
            spacer(),
            action_btn("Export", "file_text", on_click=lambda: ctrl.open_export_dialog("live"), variant="soft", key="btn-export-live"),
            action_btn("Clear", "trash", on_click=ctrl.clear_transcripts, variant="ghost", key="btn-clear-transcript"),
        ],
        left_body
    ]

    # Right Panel: Verified Grounded Suggestion Console
    sources = ctrl.retrieved_sources()
    citation_cards = []
    if sources:
        citation_cards.append(divider(color="$border"))
        citation_cards.append(
            row(align="center", gap=6)[
                box(width=6, height=6, radius="$full", bg="#6EE7B7"),
                text("VERIFIED KNOWLEDGE SOURCES (CHROMADB TOP-3):", size=10, weight=750, color="$muted", letter_spacing=0.8),
            ]
        )
        for idx, s in enumerate(sources):
            citation_cards.append(
                card(padding=(10, 12), bg="$background", border_width=1, border_color="$border", radius=8)[
                    row(gap=8, align="center")[
                        pill(f"Chunk #{idx+1}", "#A5B4FC"),
                        text(s, size=11, color="$muted", max_lines=2, grow=1),
                    ]
                ]
            )

    copy_label, copy_icon = ctrl.copy_btn_state()

    right_panel = card(
        grow=1, width=0, padding=18, gap=14, bg="$surface",
        border_width=1.5, border_color="#6EE7B7", radius=14,
        shadow=(0, 6, 20, "$glow")
    )[
        row(align="center", gap=10)[
            pill("GROUNDED SUGGESTION", "#6EE7B7", solid=True),
            text("What to say aloud:", size=13, weight=600, color="$muted"),
            spacer(),
            action_btn(
                copy_label,
                copy_icon,
                on_click=ctrl.copy_suggestion_to_clipboard,
                variant="soft",
                key="btn-copy-sugg",
            ),
        ],
        scroll(key="suggestion-scroll-area", grow=1, gap=12)[
            card(padding=18, bg="$background", border_width=1, border_color="$border", radius=12)[
                text(ctrl.suggestion_text(), size=15, weight=600, color="$text", line_height=1.5)
            ],
            *citation_cards
        ]
    ]

    return column(gap=16, grow=1, height="100%")[
        metrics,
        row(gap=16, grow=1, height=0)[left_panel, right_panel],
    ]


# --- PAGE 2: CONTEXT & PROFILES MANAGER ---
def view_profiles() -> Element:
    """Builds the profile and ChatGPT context management view."""
    chunks = ctrl.profile_chunks()

    chunk_cards = []
    if not chunks:
        chunk_cards.append(
            card(padding=36, gap=12, bg="$surface", align="center", justify="center", grow=1, radius=12)[
                icon_box("brain", "#A5B4FC", size=48, icon_size=24),
                text("No Memories Indexed for this Profile", size=15, weight=700, color="$muted"),
                text("Click [Import Context] to parse your ChatGPT conversation export (JSON, ZIP, or Markdown).", size=12.5, color="$muted"),
            ]
        )
    else:
        for idx, ch in enumerate(chunks):
            chunk_cards.append(
                card(padding=12, gap=6, bg="$surface", border_width=1, border_color="$border", radius=10)[
                    row(align="center", gap=8)[
                        pill(f"Fragment #{idx+1}", "#A5B4FC"),
                        text(f"Profile: {ctrl.selected_profile()}", size=11, color="$muted"),
                    ],
                    text(ch, size=13, color="$text", line_height=1.4)
                ]
            )

    return column(gap=16, grow=1, height="100%")[
        card(padding=18, gap=14, bg="$surface", border_width=1, border_color="$border", radius=14)[
            row(align="center", gap=12)[
                column(gap=2)[
                    text("Workspace Knowledge Profiles", size=16, weight=750, color="$text"),
                    text("Each profile maintains an isolated ChromaDB vector database.", size=12, color="$muted"),
                ],
                spacer(),
                text("Profile:", size=13, weight=600, color="$muted"),
                select(ctrl.selected_profile, ctrl.profiles_options(), key="sel-prof-page", width=190),
                action_btn("New Profile", "plus", on_click=lambda: ctrl.show_new_profile_dialog.set(True), variant="soft", key="btn-new-prof-page"),
                action_btn("Import Context", "upload", on_click=ctrl.import_context_file, variant="mint", key="btn-import-page"),
                action_btn("Reset Memory", "trash", on_click=lambda: ctrl.show_reset_dialog.set(True), variant="ghost", key="btn-reset-page"),
            ]
        ],
        card(grow=1, height=0, padding=18, gap=12, bg="$surface", border_width=1, border_color="$border", radius=14)[
            row(align="center", gap=10)[
                icon_box("layers", "#A5B4FC", size=26, icon_size=13),
                text("Indexed Knowledge Fragments Browser", size=15, weight=750, color="$text"),
                badge(f"{len(chunks)} preview items", tone="primary"),
                spacer(),
                text("Storage: ~/.context_assistant/vector_db", size=11, color="$muted"),
            ],
            scroll(key="chunks-browser-scroll", grow=1, gap=10)[chunk_cards]
        ]
    ]


# --- PAGE 3: POST-CALL REVIEW PACKETS ---
def view_packets() -> Element:
    """Builds the post-call sync packets reviewer."""
    packets = ctrl.review_packets_list()

    list_items = []
    if not packets:
        list_items.append(
            card(padding=24, gap=6, bg="$background", align="center", justify="center", radius=10)[
                text("No Review Packets Exported Yet", size=13.5, weight=600, color="$muted"),
                text("Conclude a live call session with [Stop & Save] to generate an offline sync packet.", size=11.5, color="$muted"),
            ]
        )
    else:
        for p in packets:
            is_chosen = p["name"] == ctrl.selected_packet_name()
            list_items.append(
                card(
                    key=f"packet-{p['name']}",
                    padding=12, gap=4,
                    bg="$surface_pressed" if is_chosen else "$background",
                    border_width=1,
                    border_color="$primary" if is_chosen else "$border",
                    radius=10,
                    on_click=lambda p_path=p["path"], p_name=p["name"]: ctrl.select_packet(p_path, p_name),
                )[
                    row(align="center", gap=8)[
                        pill("Packet", "#6EE7B7"),
                        text(p["time"], size=12, weight=600, color="$text"),
                        spacer(),
                        text(p["size"], size=10, color="$muted"),
                    ],
                    text(p["name"], size=11, color="$muted", max_lines=1)
                ]
            )

    content = ctrl.selected_packet_content()
    copy_label, copy_icon = ctrl.packet_copy_state()

    return row(gap=16, grow=1, height="100%")[
        card(width=320, padding=16, gap=12, bg="$surface", border_width=1, border_color="$border", radius=14, height="100%")[
            row(align="center", gap=8)[
                icon_box("file_text", "#6EE7B7", size=26, icon_size=13),
                text("Saved Packets", size=15, weight=750, color="$text"),
                badge(str(len(packets)), tone="accent"),
                spacer(),
                action_btn("Refresh", "refresh", on_click=ctrl.refresh_review_packets, variant="ghost", key="btn-refresh-packets"),
            ],
            scroll(key="packet-list-scroll", grow=1, gap=8)[list_items]
        ],
        card(grow=1, width=0, padding=18, gap=14, bg="$surface", border_width=1, border_color="$border", radius=14, height="100%")[
            row(align="center", gap=10)[
                icon_box("file_text", "#A5B4FC", size=26, icon_size=13),
                text("Packet Inspector & ChatGPT Sync", size=15, weight=750, color="$text"),
                text(ctrl.selected_packet_name(), size=12, color="$muted"),
                spacer(),
                action_btn("Export", "file_text", on_click=lambda: ctrl.open_export_dialog("packet"), variant="soft", key="btn-export-packet"),
                action_btn(
                    copy_label,
                    copy_icon,
                    on_click=ctrl.copy_packet_sync_prompt,
                    variant="primary",
                    key="btn-copy-packet-prompt",
                ),
            ],
            scroll(key="packet-content-scroll", grow=1)[
                card(padding=18, bg="$background", border_width=1, border_color="$border", radius=12)[
                    text(content if content else "Select a review packet from the left to inspect.", size=12, color="$text", line_height=1.45)
                ]
            ]
        ]
    ]


# --- PAGE 4: SETTINGS & HARDWARE DIAGNOSTICS ---
def view_settings() -> Element:
    """Builds settings panel with full Ollama/OpenAI provider & model discovery controls."""
    current_prov = ctrl.llm_provider()

    # Dynamic Provider Endpoint Field
    endpoint_fields = []
    if current_prov == "ollama":
        endpoint_fields.append(
            row(align="center", gap=14)[
                column(gap=3, grow=1)[
                    text("Ollama Server URL", size=13.5, weight=600, color="$text"),
                    text("Local default is http://localhost:11434 (supports LAN/remote Ollama IP).", size=12, color="$muted"),
                ],
                text_input(ctrl.ollama_url_input, placeholder="http://localhost:11434", key="input-ollama-url", width=300),
            ]
        )
    elif current_prov == "openai":
        endpoint_fields.append(
            row(align="center", gap=14)[
                column(gap=3, grow=1)[
                    text("OpenAI API Key", size=13.5, weight=600, color="$text"),
                    text("Official OpenAI API (https://api.openai.com/v1). Stored locally in ~/.context_assistant/config.json.", size=12, color="$muted"),
                ],
                text_input(ctrl.openai_key_input, placeholder="sk-proj-...", key="input-openai-key", width=300),
            ]
        )
    elif current_prov == "custom":
        endpoint_fields.append(
            column(gap=10)[
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("API Base URL", size=13.5, weight=600, color="$text"),
                        text("OpenAI-compatible endpoint (e.g. LM Studio, vLLM, OpenRouter, Groq).", size=12, color="$muted"),
                    ],
                    text_input(ctrl.custom_url_input, placeholder="http://localhost:1234/v1", key="input-custom-url", width=300),
                ],
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("API Key / Bearer Token", size=13.5, weight=600, color="$text"),
                        text("Optional API token for authentication.", size=12, color="$muted"),
                    ],
                    text_input(ctrl.custom_key_input, placeholder="Bearer token or leave blank", key="input-custom-key", width=300),
                ]
            ]
        )

    provider_options = [
        ("ollama", "Ollama (Local / Remote)"),
        ("openai", "OpenAI (Official ChatGPT)"),
        ("custom", "Custom OpenAI-Compatible"),
    ]

    return scroll(key="settings-scroll-area", grow=1, height="100%")[
        column(gap=16, max_width=900, padding=(0, 0, 24, 0))[
            # LLM Provider & Model Discovery Card
            card(padding=20, gap=14, bg="$surface", border_width=1, border_color="$border", radius=14)[
                row(align="center", gap=10)[
                    icon_box("cpu", "#A5B4FC", size=28, icon_size=14),
                    text("LLM Provider & Remote Model Discovery", size=16, weight=750, color="$text"),
                    spacer(),
                    pill(ctrl.engine_model_label(), "#A5B4FC"),
                ],
                text("Select an LLM provider, configure the URL/endpoint, and discover available remote models.", size=12.5, color="$muted"),
                divider(color="$border"),

                # Provider Row
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("Inference Service Provider", size=14, weight=600, color="$text"),
                        text("Choose between Ollama, OpenAI, or any OpenAI-compatible server.", size=12, color="$muted"),
                    ],
                    select(ctrl.llm_provider, provider_options, key="sel-llm-provider", width=280),
                ],

                divider(color="$border"),
                *endpoint_fields,

                divider(color="$border"),
                # Model Discovery & Selection Row
                row(align="center", gap=12)[
                    action_btn("Discover & Fetch Remote Models", "refresh", on_click=ctrl.fetch_available_models, variant="primary", key="btn-fetch-models"),
                    spacer(),
                    text(ctrl.model_status_note(), size=12, color="$muted"),
                ],

                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("Active Model for Grounded Inference", size=14, weight=600, color="$text"),
                        text("Select the model to use from the discovered models list endpoint.", size=12, color="$muted"),
                    ],
                    select(ctrl.selected_model, ctrl.available_models(), key="sel-active-model", width=300),
                    action_btn("Apply & Save", "save", on_click=ctrl.apply_model_config, variant="mint", key="btn-apply-model"),
                ]
            ],

            # System Prompt & Persona Card
            card(padding=20, gap=14, bg="$surface", border_width=1, border_color="$border", radius=14)[
                row(align="center", gap=10)[
                    icon_box("brain", "#A5B4FC", size=28, icon_size=14),
                    text("System Prompt & Agent Persona", size=16, weight=750, color="$text"),
                    spacer(),
                    pill("Customizable", "#A5B4FC"),
                ],
                text("Configure the foundational instructions given to the LLM during live calls. Use {context} as the placeholder for verified ChromaDB facts.", size=12.5, color="$muted"),
                divider(color="$border"),
                textarea(ctrl.system_prompt, rows=8, key="textarea-system-prompt", placeholder="Enter system prompt template..."),
                row(align="center", gap=16)[
                    row(align="center", gap=8)[
                        column(gap=2)[
                            text("Temperature", size=12.5, weight=600, color="$text"),
                            text("0.0 = deterministic", size=10.5, color="$muted"),
                        ],
                        text_input(ctrl.temperature, placeholder="0.0", key="input-temperature", width=75),
                    ],
                    row(align="center", gap=8)[
                        column(gap=2)[
                            text("Max Tokens", size=12.5, weight=600, color="$text"),
                            text("Response length", size=10.5, color="$muted"),
                        ],
                        text_input(ctrl.max_tokens, placeholder="150", key="input-max-tokens", width=85),
                    ],
                    row(align="center", gap=8)[
                        column(gap=2)[
                            text("VAD Silence", size=12.5, weight=600, color="$text"),
                            text("Energy threshold", size=10.5, color="$muted"),
                        ],
                        text_input(ctrl.silence_threshold, placeholder="0.015", key="input-silence-threshold", width=85),
                    ],
                ],
                divider(color="$border"),
                row(align="center", gap=10)[
                    action_btn("Save System Prompt & Parameters", "save", on_click=ctrl.save_prompt_and_params, variant="mint", key="btn-save-prompt"),
                    spacer(),
                    action_btn("Reset Prompt to Default", "rotate_ccw", on_click=ctrl.reset_system_prompt_to_default, variant="soft", key="btn-reset-prompt"),
                ]
            ],

            # Configuration Management & Factory Defaults Card
            card(padding=20, gap=14, bg="$surface", border_width=1, border_color="$border", radius=14)[
                row(align="center", gap=10)[
                    icon_box("settings", "#FDA4AF", size=28, icon_size=14),
                    text("Configuration Management & Defaults", size=16, weight=750, color="$text"),
                    spacer(),
                    pill("Persisted", "#FDA4AF"),
                ],
                text("All application settings survive across restarts in ~/.context_assistant/config.json. Restore factory defaults at any time.", size=12.5, color="$muted"),
                divider(color="$border"),
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("Restore Application Defaults", size=14, weight=600, color="$text"),
                        text("Resets system prompt, endpoints, generation parameters, and provider to factory defaults.", size=12, color="$muted"),
                    ],
                    action_btn("Restore All Defaults", "alert_triangle", on_click=ctrl.reset_all_to_defaults, variant="danger", key="btn-reset-all-defaults"),
                ]
            ],

            # Audio Hardware Card
            card(padding=20, gap=14, bg="$surface", border_width=1, border_color="$border", radius=14)[
                row(align="center", gap=10)[
                    icon_box("headphones", "#7DD3FC", size=28, icon_size=14),
                    text("Audio Hardware Configuration", size=16, weight=750, color="$text"),
                ],
                text("Choose the microphone that hears you, and the device that hears the other people in the meeting.", size=12.5, color="$muted"),
                divider(color="$border"),
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("Your microphone", size=14, weight=600, color="$text"),
                        text("Local input. Auto uses the system default microphone.", size=12, color="$muted"),
                    ],
                    select(ctrl.selected_mic, ctrl.mic_options(), key="sel-mic-settings", width=280),
                ],
                divider(color="$border"),
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("Meeting audio", size=14, weight=600, color="$text"),
                        text("Includes Multi-Output Devices. Those are recorded through the BlackHole member inside them.", size=12, color="$muted"),
                    ],
                    select(ctrl.selected_meeting, ctrl.meeting_options(), key="sel-meeting-settings", width=280),
                ],
                divider(color="$border"),
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("Spoken Language", size=14, weight=600, color="$text"),
                        text("Choose call language or Auto-Detect. Adapts available models to English or Multilingual.", size=12, color="$muted"),
                    ],
                    select(ctrl.stt_language, ctrl.language_options, key="sel-stt-lang", width=260),
                ],
                divider(color="$border"),
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("STT Speech-to-Text Quality & Model", size=14, weight=600, color="$text"),
                        text("Models dynamically adapted for the selected language (e.g. Distil-Small, Small, Distil-Large).", size=12, color="$muted"),
                    ],
                    select(ctrl.whisper_model_option, ctrl.whisper_model_choices(), key="sel-whisper-model", width=340),
                ],
                divider(color="$border"),
                row(align="center", gap=14)[
                    column(gap=3, grow=1)[
                        text("Audio Hardware VU-Meter Diagnostic", size=14, weight=600, color="$text"),
                        text("Confirm microphone and system loopback levels in the terminal.", size=12, color="$muted"),
                    ],
                    action_btn("Run Diagnostics (CLI)", "cpu", on_click=lambda: ctrl.notify("Run in terminal: ./setup.sh --test", tone="$primary"), variant="soft", key="btn-run-diagnostics"),
                ]
            ],

            # Operational Invariants Card
            card(padding=20, gap=14, bg="$surface", border_width=1, border_color="$border", radius=14)[
                row(align="center", gap=10)[
                    icon_box("shield_check", "#86EFAC", size=28, icon_size=14),
                    text("Zero-Hallucination Operational Protocol", size=16, weight=750, color="$text"),
                    spacer(),
                    pill("ENFORCED", "#86EFAC", solid=True),
                ],
                text("Immutable Invariant: If historical context lacks the answer, the assistant strictly refuses speculation and outputs the standard follow-up statement.", size=12.5, color="$muted"),
            ]
        ]
    ]


# --- PAGE ROUTER DICTIONARY ---
PAGE_VIEWS = {
    "copilot": view_copilot,
    "profiles": view_profiles,
    "packets": view_packets,
    "settings": view_settings,
}

PAGE_TITLES = {
    "copilot": ("Live Call Copilot", "Real-time audio transcription and verified ChatGPT work history grounding"),
    "profiles": ("Client Workspaces & Knowledge Base", "Browse indexed ChatGPT conversation fragments and manage client profiles"),
    "packets": ("Post-Call Review Packets", "Inspect captured dialogues and copy sync prompts for ChatGPT Plus workspace memory"),
    "settings": ("System Settings & Diagnostics", "LLM provider endpoints, remote model discovery, and audio routing"),
}


# --- ROOT DECLARATIVE VIEW ---
def build_view() -> Element:
    """Builds application window layout with sidebar, header, dynamic pages, and overlays."""
    current_key = ctrl.active_page()
    title, subtitle = PAGE_TITLES.get(current_key, ("Live Copilot", ""))

    # Session Toggle Button (Start vs Stop & Save)
    if ctrl.is_listening():
        session_btn = action_btn(
            "Stop & Save",
            "square",
            on_click=ctrl.stop_session,
            variant="danger",
            key="btn-stop-session",
        )
    else:
        session_btn = action_btn(
            "Start Session",
            "play",
            on_click=ctrl.start_session,
            variant="primary",
            key="btn-start-session",
        )

    # Top Header Controls Strip (3-column layout: Left Title, Centered Status, Right Controls)
    header_strip = row(align="center", padding=(20, 28, 14, 28))[
        # Left: Page Title & Subtitle
        row(width=340, align="center")[
            column(gap=3)[
                text(title, size=24, weight=750, letter_spacing=-0.5, color="$text"),
                text(subtitle, size=13, color="$muted"),
            ]
        ],
        # Center: Live Status Indicator (Centered in Header)
        row(grow=1, align="center", justify="center")[
            pill(ctrl.status_text(), ctrl.status_color(), solid=True),
        ],
        # Right: Source Pill and Toggled Session Button
        row(width=340, align="center", justify="end", gap=12)[
            pill(f"Source: {ctrl.active_device_label()}", "#94A3B8"),
            session_btn,
        ],
    ]

    # Main Content Area
    main_content = column(grow=1, width=0, bg="$background")[
        header_strip,
        box(height=2),
        column(grow=1, height=0, padding=(0, 28, 20, 28))[
            pages(ctrl.active_page, PAGE_VIEWS, key="main-page-router")
        ]
    ]

    # Modal Dialogs
    modal_new_profile = dialog(
        ctrl.show_new_profile_dialog,
        column(gap=16, padding=20)[
            text("Create Client Profile", size=16, weight=750, color="$text"),
            text("Each client has a dedicated, isolated ChromaDB vector collection.", size=13, color="$muted"),
            text_input(ctrl.new_profile_name, placeholder="e.g. Acme_Corp_Client", key="input-new-prof-name"),
            row(gap=10, justify="end")[
                action_btn("Cancel", "x", on_click=lambda: ctrl.show_new_profile_dialog.set(False), variant="soft", key="btn-cancel-new-prof"),
                action_btn("Create Profile", "plus", on_click=ctrl.create_new_profile, variant="primary", key="btn-confirm-new-prof"),
            ]
        ],
        title="New Workspace Profile",
        width=400
    )

    modal_export = dialog(
        ctrl.show_export_dialog,
        column(gap=14, padding=8)[
            text("Save the full conversation or a meeting resume.", size=13, color="$muted"),
            text("What to export", size=12, weight=700, color="$text"),
            row(gap=8)[
                action_btn(
                    "Full conversation",
                    "message",
                    on_click=lambda: ctrl.export_kind.set(KIND_CONVERSATION),
                    variant="primary" if ctrl.export_kind() == KIND_CONVERSATION else "soft",
                    key="btn-export-kind-conversation",
                ),
                action_btn(
                    "Meeting resume",
                    "list",
                    on_click=lambda: ctrl.export_kind.set(KIND_RESUME),
                    variant="primary" if ctrl.export_kind() == KIND_RESUME else "soft",
                    key="btn-export-kind-resume",
                ),
            ],
            text("File format", size=12, weight=700, color="$text"),
            row(gap=8)[
                action_btn(
                    "Markdown",
                    "file_text",
                    on_click=lambda: ctrl.export_format.set("md"),
                    variant="primary" if ctrl.export_format() == "md" else "soft",
                    key="btn-export-fmt-md",
                ),
                action_btn(
                    "Text",
                    "type",
                    on_click=lambda: ctrl.export_format.set("txt"),
                    variant="primary" if ctrl.export_format() == "txt" else "soft",
                    key="btn-export-fmt-txt",
                ),
                action_btn(
                    "CSV",
                    "table",
                    on_click=lambda: ctrl.export_format.set("csv"),
                    variant="primary" if ctrl.export_format() == "csv" else "soft",
                    key="btn-export-fmt-csv",
                ),
            ],
            row(gap=10, justify="end")[
                action_btn("Cancel", "x", on_click=lambda: ctrl.show_export_dialog.set(False), variant="soft", key="btn-export-cancel"),
                action_btn("Save file", "save", on_click=ctrl.save_call_export, variant="primary", key="btn-export-save"),
            ],
        ],
        title="Export call",
        width=480,
    )

    modal_reset_confirm = dialog(
        ctrl.show_reset_dialog,
        column(gap=16, padding=20)[
            text(f"Wipe Memory for [{ctrl.selected_profile()}]?", size=16, weight=750, color="$danger"),
            text(
                "This action permanently erases all indexed vector fragments for this profile in ChromaDB.\n"
                "This action cannot be undone.",
                size=13,
                color="$text"
            ),
            row(gap=10, justify="end")[
                action_btn("Cancel", "x", on_click=lambda: ctrl.show_reset_dialog.set(False), variant="soft", key="btn-cancel-reset"),
                action_btn("Confirm Wipe", "trash", on_click=ctrl.reset_profile_memory, variant="danger", key="btn-confirm-reset"),
            ]
        ],
        title="Confirm Reset",
        width=420
    )

    # Modal Dialog: Graceful Shutdown Progress Indicator
    modal_closing = dialog(
        ctrl.is_closing,
        column(gap=16, padding=24, align="center", bg="$surface", radius=14, border_width=1, border_color="$border")[
            row(align="center", justify="center", gap=12)[
                spinner(size=20, tone="#FDA4AF"),
                text("Closing Live Copilot", size=17, weight=750, color="$text"),
            ],
            text(
                ctrl.closing_status(),
                size=13.5,
                color="$text",
                text_align="center"
            ),
            box(height=4),
            row(gap=6, align="center")[
                box(width=6, height=6, radius="$full", bg="#FDA4AF"),
                text("Please wait while resources are safely saved and released...", size=11, color="$muted"),
            ]
        ],
        title="Application Shutdown",
        width=440
    )

    # Toast Notifications
    toast_widget = toast(ctrl.toast_msg, tone=ctrl.toast_tone(), key="toast-notification")

    return row(grow=1, bg="$background")[
        build_sidebar(),
        main_content,
        modal_new_profile,
        modal_export,
        modal_reset_confirm,
        modal_closing,
        toast_widget,
    ]


def main():
    app = App(
        build_view,
        title="Live Call Assistant — Grounded ChatGPT Copilot",
        size=(1240, 800),
        theme=ctrl.current_theme
    )
    ctrl.app = app
    app.main.on_close = ctrl.cleanup
    print("[Aurora UI] Starting desktop application event loop...")
    app.run()


if __name__ == "__main__":
    main()
