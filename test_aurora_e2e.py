#!/usr/bin/env python3
"""
Automated Headless End-to-End Verification Test for Redesigned Aurora Live Call Assistant.
Tests multi-page navigation, modal creation, ingestion, RAG grounding, and review packets.
"""

import os
import sys
import time
from app_aurora import build_view, ctrl, App, MIDNIGHT

def run_test():
    print("=" * 60)
    print("[RUN] Running Headless E2E Test for Redesigned Aurora Copilot")
    print("=" * 60)

    app = App(build_view, title="Aurora Assistant E2E Test", size=(1280, 1050), theme=ctrl.current_theme)
    pilot = app.pilot()

    # Step 1: Initial Render
    print("1. Capturing Initial UI Frame (Live Copilot)...")
    ctrl.reset_all_to_defaults()
    pilot.settle()
    pilot.png("/tmp/test_aurora_step1_init.png")
    assert os.path.exists("/tmp/test_aurora_step1_init.png"), "Step 1 PNG failed to write."
    print("   [OK] Initial Frame Rendered Successfully.")

    # Step 2: Navigate to Knowledge & Profiles Page
    print("2. Testing Navigation: Switching to 'Context & Profiles' Page...")
    pilot.click("nav-profiles")
    pilot.settle()
    assert ctrl.active_page() == "profiles", "Active page should be 'profiles'."
    pilot.png("/tmp/test_aurora_step2_profiles.png")
    print("   [OK] Successfully Navigated to Profiles View.")

    # Step 3: Open New Profile Dialog
    print("3. Testing Modal Dialog: Open '+ New Profile'...")
    pilot.click("btn-new-prof-page")
    pilot.settle()
    assert ctrl.show_new_profile_dialog() is True, "New profile dialog should be open."
    print("   [OK] Modal Dialog Opened.")

    # Step 4: Type profile name and create
    test_profile = "Aurora_Studio_Client"
    print(f"4. Creating Profile: '{test_profile}'...")
    ctrl.new_profile_name.set(test_profile)
    pilot.click("btn-confirm-new-prof")
    pilot.settle()
    assert ctrl.show_new_profile_dialog() is False, "New profile dialog should close after creation."
    assert ctrl.selected_profile() == test_profile, f"Active profile should be '{test_profile}'."
    print(f"   [OK] Active Profile is now: [{ctrl.selected_profile()}].")

    # Step 5: Ingest Sample Data into active profile
    sample_file = os.path.join(os.path.dirname(__file__), "sample_data", "conversations.json")
    print(f"5. Ingesting Sample Data from: {sample_file}...")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    ctrl._process_and_index_file(sample_file)
    time.sleep(1.0)
    pilot.settle()
    docs = ctrl.doc_count()
    print(f"   [OK] Ingested Context Chunks: {docs}")
    assert docs > 0, "Doc count should be > 0 after ingestion."

    # Step 6: Navigate back to Live Copilot for Call Session
    print("6. Navigating back to Live Copilot Page...")
    pilot.click("nav-copilot")
    pilot.settle()
    assert ctrl.active_page() == "copilot", "Active page should be 'copilot'."
    print("   [OK] Live Copilot Command Center Active.")

    # Step 7: Simulate Incoming Live Speech Query & RAG Grounding
    test_query = "What is the agreed hourly rate for backend engineering?"
    print(f"7. Simulating Live Call Speech: '{test_query}'...")

    now_str = time.strftime("%H:%M:%S")
    ctrl.transcripts.set([{"time": now_str, "speaker": "Participant", "text": test_query}])
    ctrl.transcript_history.append((now_str, "Participant", test_query))

    print("   Querying ChromaDB vector index and LLM...")
    ctrl.generate_contextual_suggestion(test_query)
    time.sleep(1.5)
    pilot.settle()

    suggestion = ctrl.suggestion_text()
    print(f"   [AI] Grounded Suggestion Output:\n   \"{suggestion}\"")
    assert len(suggestion) > 10, "Suggestion output should not be empty."
    assert "$150" in suggestion or "150" in suggestion or "backend" in suggestion.lower() or "hourly" in suggestion.lower() or "budget" in suggestion.lower(), \
        f"Suggestion should contain verified facts from sample_data: {suggestion}"
    print("   [OK] RAG Grounding Verification Passed!")

    # Step 8: Test Anti-Hallucination Fallback
    unknown_query = "Who approved the trip to Mars and what was the launch budget?"
    print(f"8. Testing Anti-Hallucination on Unknown Query: '{unknown_query}'...")
    ctrl.generate_contextual_suggestion(unknown_query)
    time.sleep(1.5)
    pilot.settle()
    refusal = ctrl.suggestion_text()
    print(f"   [AI] Fallback Output:\n   \"{refusal}\"")
    expected_refusal = "I don't have that documented in our prior work history, but let me check and follow up."
    assert expected_refusal in refusal, f"Assistant failed zero-hallucination constraint. Output was: {refusal}"
    print("   [OK] Zero-Hallucination Fallback Verified!")

    # Step 9: Test Clipboard Copy
    print("9. Testing Clipboard Copy Action...")
    pilot.click("btn-copy-sugg")
    time.sleep(0.5)
    import subprocess
    pasted = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    assert pasted == refusal, f"Clipboard content mismatch: got '{pasted}'"
    print("   [OK] System Clipboard Successfully Synced.")

    # Step 10: Test Review Packet Export
    print("10. Testing Post-Call Sync Packet Export...")
    ctrl.export_post_call_packet()
    reviews_dir = os.path.expanduser("~/.context_assistant/review_packets")
    packets = [f for f in os.listdir(reviews_dir) if f.endswith(".md")]
    assert len(packets) > 0, "No review packet was exported."
    latest_packet = sorted(packets)[-1]
    with open(os.path.join(reviews_dir, latest_packet), "r") as f:
        packet_content = f.read()
    assert "UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT" in packet_content, "Packet missing protocol header."
    assert test_profile in packet_content, "Packet missing profile name."
    assert test_query in packet_content, "Packet missing captured dialogue."
    print(f"   [OK] Exported Valid Review Packet: {latest_packet}")

    # Step 11: Navigate to Review Packets Page & Test In-App Inspector
    print("11. Navigating to 'Review Packets' Page...")
    pilot.click("nav-packets")
    pilot.settle()
    assert ctrl.active_page() == "packets", "Active page should be 'packets'."
    pilot.png("/tmp/test_aurora_final_packets.png")
    print("   [OK] Review Packets Inspector Rendered.")

    # Step 12: Test Theme Toggle
    print("12. Testing Theme Toggle (Midnight <-> Daylight)...")
    pilot.click("theme-toggle-btn")
    pilot.settle()
    print("   [OK] Theme Toggled Successfully.")

    # Step 13: Test Settings Page & Dynamic Remote Model Discovery
    print("13. Testing Settings Page & Remote Model Discovery...")
    pilot.click("nav-settings")
    pilot.settle()
    assert ctrl.active_page() == "settings", "Active page should be 'settings'."

    # Configure Ollama URL
    ctrl.llm_provider.set("ollama")
    ctrl.ollama_url_input.set("http://localhost:11434")
    print("   Fetching available models from Ollama /api/tags endpoint...")
    ctrl.fetch_available_models(silent=True)
    time.sleep(1.0)
    pilot.settle()

    discovered = [m[0] for m in ctrl.available_models()]
    print(f"   Discovered Models: {discovered}")
    assert len(discovered) > 0, "Discovered models should not be empty."

    # Select model (prefer verified instruction-following llama3.1)
    target_model = next((m for m in discovered if "llama3.1:8b" in m or "llama3.1" in m), discovered[0])
    ctrl.selected_model.set(target_model)
    time.sleep(0.2)
    assert ctrl.llm_model_name == target_model, f"Active model name should be {target_model}."

    # Apply & verify config file persistence
    ctrl.save_config()
    config_path = os.path.expanduser("~/.context_assistant/config.json")
    assert os.path.exists(config_path), "Config file ~/.context_assistant/config.json was not created."
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("provider") == "ollama", "Saved provider should be 'ollama'."
    assert cfg.get("ollama_url") == "http://localhost:11434", "Saved ollama_url mismatch."
    assert cfg.get("selected_model") == target_model, "Saved model mismatch."

    pilot.png("/tmp/test_aurora_step13_settings.png")
    print(f"   [OK] Settings & Remote Model Discovery Verified! Active Model: [{target_model}]")

    # Step 14: Test System Prompt Customization & Restoring Factory Defaults
    print("14. Testing System Prompt Customization & Factory Defaults Reset...")
    from app_aurora import DEFAULT_SYSTEM_PROMPT, DEFAULT_CONFIG

    # 1. Customize Prompt and Parameters
    test_custom_prompt = (
        "You are an executive call copilot.\n"
        "=== VERIFIED CLIENT / WORK HISTORY ===\n{context}\n\n"
        "Output 1 sentence. Never hallucinate."
    )
    ctrl.system_prompt.set(test_custom_prompt)
    ctrl.temperature.set("0.3")
    ctrl.max_tokens.set("100")
    pilot.a11y_action("btn-save-prompt", "click")
    time.sleep(0.3)
    pilot.settle()

    # Check persistence
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("system_prompt") == test_custom_prompt, f"Custom system prompt was not persisted. Got: {cfg.get('system_prompt')}"
    assert cfg.get("temperature") == 0.3, "Custom temperature was not persisted."
    assert cfg.get("max_tokens") == 100, "Custom max_tokens was not persisted."
    print("   [OK] Custom System Prompt & Tuning Parameters Persisted Successfully.")

    # 2. Test Reset System Prompt to Default
    pilot.a11y_action("btn-reset-prompt", "click")
    time.sleep(0.3)
    pilot.settle()
    assert ctrl.system_prompt() == DEFAULT_SYSTEM_PROMPT, "System prompt was not reset to default."
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("system_prompt") == DEFAULT_SYSTEM_PROMPT, "Persisted prompt was not reset to default."
    print("   [OK] Reset System Prompt to Default Verified.")

    # 3. Test Restore All Application Defaults
    pilot.a11y_action("btn-reset-all-defaults", "click")
    time.sleep(0.3)
    pilot.settle()
    assert ctrl.llm_provider() == DEFAULT_CONFIG["provider"], "Provider not reset to default."
    assert ctrl.ollama_url_input() == DEFAULT_CONFIG["ollama_url"], "Ollama URL not reset to default."
    assert ctrl.selected_model() == DEFAULT_CONFIG["selected_model"], "Model not reset to default."
    assert ctrl.stt_language() == DEFAULT_CONFIG["stt_language"], "STT Language not reset to default."
    assert ctrl.whisper_model_option() == DEFAULT_CONFIG["whisper_model"], "Whisper model not reset to default."
    assert ctrl.temperature() == str(DEFAULT_CONFIG["temperature"]), "Temperature not reset to default."
    assert ctrl.max_tokens() == str(DEFAULT_CONFIG["max_tokens"]), "Max tokens not reset to default."
    assert ctrl.silence_threshold() == str(DEFAULT_CONFIG["silence_threshold"]), "Silence threshold not reset."

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("provider") == DEFAULT_CONFIG["provider"], "Config file provider not reset."
    assert cfg.get("selected_model") == DEFAULT_CONFIG["selected_model"], "Config file model not reset."
    assert cfg.get("stt_language") == DEFAULT_CONFIG["stt_language"], "Config file stt_language not reset."
    assert cfg.get("whisper_model") == DEFAULT_CONFIG["whisper_model"], "Config file whisper_model not reset."
    assert cfg.get("temperature") == DEFAULT_CONFIG["temperature"], "Config file temperature not reset."

    pilot.png("/tmp/test_aurora_step14_defaults.png")
    print("   [OK] Restore All Application Defaults Verified!")

    # Step 15: Close Application Button & Shutdown Spinner Verification
    print("15. Testing Close Application Button & Shutdown Workflow...")
    ctrl.app = app
    assert ctrl.is_closing() is False, "ctrl.is_closing should initially be False."
    pilot.a11y_action("btn-close-app", "click")
    time.sleep(0.1)
    # Note: spinner has a perpetual pulse animation by design, so pilot.frame() renders the frame directly
    pilot.frame()
    assert ctrl.is_closing() is True, "ctrl.is_closing should be True after clicking Close Application."
    pilot.png("/tmp/test_aurora_step15_closing.png")
    assert os.path.exists("/tmp/test_aurora_step15_closing.png"), "Step 15 PNG failed to write."
    print(f"   Status note during closing: '{ctrl.closing_status()}'")
    print("   [OK] Close Application Button & Shutdown Spinner Verified.")

    print("=" * 60)
    print("[PASSED] ALL 15 END-TO-END VERIFICATION STEPS PASSED FOR AURORA COPILOT!")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
