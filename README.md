# Live Call AI Assistant — ChatGPT Grounded Copilot

A Python application for macOS that listens during live business and work calls, accurately follows the conversation, and provides real-time, concise, natural responses you can speak aloud based **strictly** on the verified context from your ChatGPT work conversations.

---

## Key Features

1. **Grounded Work Context (No Hallucinations)**:
   - Segregated ChromaDB vector database per client/workspace profile.
   - Answers strictly use verified facts, agreements, dates, and SLA terms from imported ChatGPT work conversations.
   - If a question asks about unrecorded information, the assistant explicitly states: *"I don't have that documented in our prior work history, but let me check and follow up."*

2. **Dual-Channel Audio Capture**:
   - Captures both the remote caller (system audio via BlackHole loopback) and you (microphone), even when wearing headphones or AirPods.
   - Dynamic sample rate detection handles Bluetooth SCO rate switches (16 kHz vs 44.1/48 kHz) with automatic 16 kHz resampling.

3. **Ultra-Fast, Multilingual Local Speech-to-Text**:
   - Powered by `faster-whisper` with **`small.en`** as the default. A sentence transcribes in about a second, and distilled models are available when you want the fastest option.
   - **Multilingual Support & Language Selector**: Choose between English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Korean, or Auto-Detect.
   - **Dynamic Model Adaptation**:
     - *English selected*: Uses high-speed English-specialized models (**Distil Small**, **Small.en**, **Base.en**, **Distil Medium**, **Medium.en**).
     - *Other languages / Auto-Detect selected*: Automatically adapts dropdown to compatible Multilingual Whisper models (**Small Multilingual**, **Base Multilingual**, **Medium Multilingual**, **Distil Large v3**, **Large v3**).
   - **Responsive Trailing-Window VAD**: Detects a natural 500ms pause at the end of speech, dispatching phrases immediately without lingering silence delays.
   - **Audio Amplitude Normalization**: Rescales input audio to nominal 95% peak volume before inference, eliminating issues from low mic gain or muffled voices.
   - **Fast Accurate Decoding**: Beam search (`beam_size=5`, `temperature=0.0`) on phrases the energy detector already trimmed, without a second VAD pass (`condition_on_previous_text=False`).
   - Zero external cloud latency or third-party audio streaming.

4. **Configurable LLM Provider & Remote Model Discovery**:
   - Supports **Ollama** (configurable local URL `http://localhost:11434`, LAN IP, or remote host).
   - Supports official **OpenAI** (`gpt-4o`, `gpt-4o-mini`, `o1-mini`) with custom API key.
   - Supports **Any OpenAI-Compatible Engine** (LM Studio, vLLM, OpenRouter, Groq, Together) with custom base URL and token.
   - **Dynamic Model Endpoint Discovery**: Automatically queries the remote endpoint (`/api/tags` or `/v1/models`) and populates an interactive dropdown with discovered models.

5. **Customizable System Prompt & Generation Parameters**:
   - Live editable System Prompt template with `{context}` substitution for verified ChromaDB facts.
   - Tunable generation parameters: `temperature` (default `0.0`), `max_tokens` (default `150`), and VAD `silence_threshold` (default `0.015`).
   - Built-in **"Reset Prompt to Default"** and **"Restore All Defaults"** to return to standard defaults at any time.
   - All options persist locally in `~/.context_assistant/config.json` across restarts.

6. **Post-Call Review Packet**:
   - Automatically generates a structured Markdown review packet upon stopping a call.
   - Includes formatted conversation dialogue and a copy-pasteable ChatGPT ingestion prompt (`UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT`) to sync learnings back into your ChatGPT thread.

7. **Graceful Application Shutdown & Spinner**:
   - Dedicated "Close Application" button located at the bottom of the left navigation drawer.
   - Displays real-time spinner animations and status notes during shutdown (gracefully concluding active audio sessions, exporting review packets, persisting configurations, and releasing hardware hooks).

---

## Available Editions

The application is available in two desktop interface editions:
1. **Aurora Edition (`app_aurora.py`) [Recommended]**:
   - Built with [Aurora UI](https://github.com/YoshiLoL0526/aurora) — a declarative, reactive desktop framework with a high-performance native Rust core, Google Skia rendering engine, and Taffy flexbox layout.
   - Includes real-time Signals, reactive modal dialogs, floating toast notifications, one-click clipboard copy, and visual citation panels.
2. **Elevated Tkinter Edition (`app.py`)**:
   - Modernized multi-page desktop experience with Midnight & Daylight themes, 4-page navigation drawer, descending dialogue stream with latest badge, verified ChromaDB citation cards, review packets inspector, dynamic model discovery, and graceful shutdown animation.

---

## Directory Structure

```text
ai_assistant/
├── app_aurora.py              # Aurora declarative desktop UI (Rust/Skia engine)
├── app.py                     # Elevated Tkinter multi-page desktop application
├── setup.sh                   # Automated macOS launcher & environment checker
├── test_aurora_e2e.py         # 15-step headless E2E verification test for Aurora UI
├── test_tkinter_e2e.py        # 11-step headless E2E verification test for Tkinter app
├── test_workflow.py           # Headless automated verification test for RAG/LLM
├── test_audio.py              # Real-time ASCII VU-meter audio diagnostics utility
├── requirements.txt           # Default Tkinter dependencies (sounddevice, faster-whisper, chromadb, pillow)
├── requirements_aurora.txt    # Aurora UI dependencies (+ aurora-ui Rust/Skia engine)
├── sample_data/
│   └── conversations.json     # Sample ChatGPT export for testing Project Alpha & SLAs
└── README.md                  # This documentation
```

Application sandbox files are stored in `~/.context_assistant/`:
- `~/.context_assistant/profiles/` — Client workspaces.
- `~/.context_assistant/vector_db/` — ChromaDB vector database.
- `~/.context_assistant/review_packets/` — Exported post-call summary packets.

---

## Quick Start

### 1. Launch with One Command

Make sure [Ollama](https://ollama.com) is installed and has Llama 3.1:
```bash
ollama pull llama3.1
```

Then run the setup launcher (launches Aurora by default):
```bash
./setup.sh
```

**Alternative Launch Modes:**
- Run Aurora UI directly: `python3 app_aurora.py` or `./setup.sh`
- Run Classic Tkinter: `./setup.sh --tkinter`
- Run Audio Signal Diagnostics: `./setup.sh --test`
- Run Headless Aurora E2E Test: `./setup.sh --test-e2e`

*(Optional: To use OpenAI API instead of Ollama, export your key: `export OPENAI_API_KEY="sk-..."` before launching).*


---

## Audio Setup for Live Calls (Headphones + Microphone)

To allow the assistant to hear both participants during a live call while you wear headphones:

### 1. Install BlackHole Loopback Driver
```bash
brew install blackhole-2ch
```

### 2. Configure macOS Multi-Output Device
1. Open **Audio MIDI Setup** on your Mac (press `Cmd + Space` and search for "Audio MIDI Setup").
2. Click the **`+`** button in the lower-left corner and select **Create Multi-Output Device**.
3. Check both:
   - **Your Headphones / AirPods** (set as Master Device).
   - **BlackHole 2ch**.
4. In macOS **System Settings / Sound**:
   - Set **Output** to your new **Multi-Output Device**.
5. In your meeting app (Zoom, Google Meet, Teams, Slack):
   - **Speaker / Audio Output**: Select **Multi-Output Device** (or System Default).
   - **Microphone**: Select your regular **Microphone**.
6. In **Live Call Assistant**:
   - Audio input dropdown automatically detects and selects **BlackHole 2ch**.

---

## Verifying Audio with Diagnostics

Run the signal meter and choose the input to check: your microphone, BlackHole, or any other listed device.

```bash
./setup.sh --test
# or
python3 test_audio.py              # prompts for a device
python3 test_audio.py mic          # default microphone
python3 test_audio.py blackhole    # BlackHole loopback
python3 test_audio.py 2            # PortAudio device index
```

Speak into the mic, or play sound through the selected loopback. The terminal shows a real-time VU meter:
```text
Audio Level: [====================                    ] 0.0412 (Peak: 0.1284)
```

---

## How to Use the App

### Step 1: Manage Profiles
- In the top bar or sidebar, select an existing workspace profile or click **`+ New Profile`** to create one (e.g. `Alpha_Client`).

### Step 2: Ingest Context
- Click **`Import Context`**.
- Select a ChatGPT export file (`conversations.json`, account export `.zip`, `.md`, or `.txt`).
- You can test immediately using `sample_data/conversations.json`.
- The assistant indexes the dialogue into ChromaDB and reports the total fragment count.

### Step 3: Run the Live Call
- Click **`Start Session`**.
- As participants speak, the transcript appears in the **Live Call Dialogue Stream** (left panel).
- Grounded, concise response recommendations appear in green in the **Verified Grounded Suggestion** (right panel).

### Step 4: Stop and Export
- Click **`Stop & Save`**.
- A notification will show the location of the newly generated review packet:
  `~/.context_assistant/review_packets/review_packet_<timestamp>.md`.
- Copy the formatted block into your main ChatGPT thread to keep its memory synced.

---

## Running Verification Tests

To verify that vector search, Ollama grounded inference, and hallucination prevention are working without starting audio:
```bash
python3 test_workflow.py
```
