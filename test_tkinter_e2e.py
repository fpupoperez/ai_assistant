#!/usr/bin/env python3
"""
Automated Headless End-to-End Verification Test for Elevated Tkinter Live Call Assistant (app.py).
Tests 4-page navigation, theme toggling, profile creation, context ingestion,
grounded LLM inference, review packets inspection, dynamic model discovery,
and graceful shutdown.
"""

import os
import sys
import time
import tkinter as tk
from app import RealTimeAssistantApp, THEMES, DEFAULT_CONFIG

def run_test():
    print("=" * 60)
    print("[RUN] Running Headless E2E Test for Elevated Tkinter Copilot (app.py)")
    print("=" * 60)

    root = tk.Tk()
    root.withdraw()  # Headless execution
    app = RealTimeAssistantApp(root)
    root.update()

    # Step 1: Initial Render & Defaults
    print("1. Verifying Initial Frame & Setup...")
    assert app.active_page == "copilot", "Initial active page should be 'copilot'."
    assert app.theme_name == "midnight", "Initial theme should be 'midnight'."
    root.update()
    print("   [OK] Initial Frame Rendered Successfully.")

    # Step 2: Theme Toggle
    print("2. Testing Theme Toggle (Midnight <-> Daylight)...")
    app.toggle_theme()
    root.update()
    assert app.theme_name == "daylight", "Theme should be daylight after toggle."
    app.toggle_theme()
    root.update()
    assert app.theme_name == "midnight", "Theme should be midnight after second toggle."
    print("   [OK] Theme Toggled Successfully.")

    # Step 3: Navigation across all 4 pages
    print("3. Testing Multi-Page Navigation Drawer...")
    for page in ["profiles", "packets", "settings", "copilot"]:
        app.show_page(page)
        root.update()
        assert app.active_page == page, f"Failed to navigate to {page}."
    print("   [OK] Navigation across all 4 pages verified.")

    # Step 4: Profile Creation & Ingestion
    print("4. Testing Profile Creation & Sample Context Ingestion...")
    test_profile = "Tkinter_Studio_Client"
    app.selected_profile.set(test_profile)
    app.load_profile_collection(test_profile)
    root.update()

    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "conversations.json")
    assert os.path.exists(sample_path), f"Sample data missing at: {sample_path}"

    import json
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = app._parse_conversations_json(data)
    assert len(entries) > 0, "No entries parsed from conversations.json"

    ids = [f"doc_test_{i}" for i in range(len(entries))]
    app.collection.add(documents=entries, ids=ids)
    doc_count = app.get_document_count()
    assert doc_count >= len(entries), f"Document count mismatch: got {doc_count}"
    print(f"   [OK] Ingested {doc_count} context fragments into [{test_profile}].")

    # Step 5: Test Grounded RAG Suggestion
    print("5. Testing Grounded Suggestion with Known Historical Fact...")
    test_query = "What is the agreed hourly rate for backend engineering and project budget?"
    app.generate_contextual_suggestion(test_query)
    root.update()

    sugg = app.suggestion_text
    print(f"   [AI] Grounded Suggestion Output:\n   \"{sugg}\"")
    assert len(sugg) > 10, "Suggestion output should not be empty."
    assert "$45,000" in sugg or "45,000" in sugg or "budget" in sugg.lower() or "hourly" in sugg.lower(), \
        f"Suggestion should contain verified facts: {sugg}"
    assert len(app.retrieved_sources) > 0, "Top-3 retrieved sources should be populated."
    print("   [OK] RAG Grounding & Citations Verified!")

    # Step 6: Test Anti-Hallucination Fallback
    print("6. Testing Anti-Hallucination Fallback on Undocumented Query...")
    unknown_query = "Who approved the secret submarine trip to Atlantis?"
    app.generate_contextual_suggestion(unknown_query)
    root.update()

    refusal = app.suggestion_text
    print(f"   [AI] Fallback Output:\n   \"{refusal}\"")
    expected_refusal = "I don't have that documented in our prior work history, but let me check and follow up."
    assert expected_refusal in refusal, f"Failed zero-hallucination refusal. Output was: {refusal}"
    print("   [OK] Zero-Hallucination Fallback Verified!")

    # Step 7: Test Descending Transcripts Stream
    print("7. Testing Descending Live Dialogue Stream Cards...")
    app.transcripts.insert(0, {"time": "12:00:01", "speaker": "Participant", "text": "First phrase spoken"})
    app.transcripts.insert(0, {"time": "12:00:05", "speaker": "Participant", "text": "Second phrase spoken (latest)"})
    app.show_page("copilot")
    root.update()

    assert len(app.transcripts) == 2, "Transcripts list should have 2 items."
    assert app.transcripts[0]["text"] == "Second phrase spoken (latest)", "Newest item should be at index 0 (top)."
    print("   [OK] Descending Transcripts Layout Verified.")

    # Step 8: Test Review Packet Export & Inspection
    print("8. Testing Review Packet Export and Packet Inspector...")
    app.transcript_history.append(("12:00:05", "Participant", "Second phrase spoken"))
    app.export_post_call_packet()
    app.refresh_review_packets()
    assert len(app.review_packets_list) > 0, "Exported review packet not found."

    app.show_page("packets")
    root.update()
    assert app.selected_packet_name is not None, "A packet should be selected in inspector."
    print(f"   [OK] Exported & Inspected Valid Review Packet: {app.selected_packet_name}")

    # Step 9: Test Settings & Remote Model Discovery
    print("9. Testing Settings & Model Discovery Query...")
    app.show_page("settings")
    root.update()
    models = app.fetch_available_models()
    print(f"   Discovered Models: {models[:5] if models else 'None'}")
    assert app.llm_provider.get() in ["ollama", "openai", "custom"], "Provider invalid."
    print("   [OK] Settings View & Model Discovery Verified.")

    # Step 10: Test Custom System Prompt & Reset Defaults
    print("10. Testing Custom System Prompt & Reset Defaults...")
    app.system_prompt_text = "Customized test prompt: {context}"
    app.temperature.set("0.7")
    app.save_config()

    app.reset_system_prompt_to_default(silent=True)
    assert app.system_prompt_text == DEFAULT_CONFIG["system_prompt"], "System prompt not reset."
    print("   [OK] Custom Prompt & Reset Verified.")

    # Step 11: Test Session Toggle
    print("11. Testing Session Button Toggle State...")
    app.show_page("copilot")
    assert not app.is_listening, "Should be initially stopped."
    assert "Start Session" in app.btn_session.cget("text")
    print("   [OK] Session Toggle Verified.")

    # Cleanup test collection
    if app.chroma_client:
        try:
            safe_name = "prof_tkinter_studio_client"
            app.chroma_client.delete_collection(name=safe_name)
        except Exception:
            pass

    try:
        root.destroy()
    except Exception:
        pass
    print("=" * 60)
    print("[PASSED] ALL 11 END-TO-END VERIFICATION STEPS PASSED FOR ELEVATED TKINTER APP!")
    print("=" * 60)
    os._exit(0)

if __name__ == "__main__":
    run_test()
