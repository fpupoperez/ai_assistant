#!/usr/bin/env python3
"""
Personal Real-Time AI Live Call Assistant -- macOS Tkinter Desktop Edition
Listens during live calls (mic + loopback), transcribes speech locally in real-time,
queries verified work history via ChromaDB vector embeddings, and displays concise,
grounded suggestions to speak aloud.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Tuple

# Suppress benign multiprocessing resource_tracker cleanup warning on shutdown
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

import chromadb
from faster_whisper import WhisperModel
import numpy as np
from openai import OpenAI
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
from ui_icons import get_tk_icon

# --- APPLICATION SANDBOX DIRECTORIES ---
BASE_DIR = os.path.expanduser("~/.context_assistant")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
REVIEWS_DIR = os.path.join(BASE_DIR, "review_packets")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

for directory in [BASE_DIR, PROFILES_DIR, VECTOR_DB_DIR, REVIEWS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- MODEL AND LANGUAGE DEFINITIONS ---
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
THEMES = {
    "midnight": {
        "bg": "#0F1219",
        "surface": "#181D2A",
        "surface_alt": "#22293B",
        "border": "#283147",
        "text": "#F1F5F9",
        "text_bright": "#FFFFFF",
        "muted": "#94A3B8",
        "primary": "#818CF8",
        "primary_hover": "#6366F1",
        "primary_light": "#312E81",
        "accent": "#818CF8",
        "success": "#34D399",
        "success_bg": "#064E3B",
        "danger": "#FDA4AF",
        "danger_bg": "#4C0519",
        "danger_btn": "#F43F5E",
        "warning": "#FCD34D",
        "card_bg": "#181D2A",
        "pill_bg": "#22293B",
        "input_bg": "#0F1219",
        "scroll_trough": "#181D2A",
        "scroll_thumb": "#334155",
    },
    "daylight": {
        "bg": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface_alt": "#F1F5F9",
        "border": "#E2E8F0",
        "text": "#1E293B",
        "text_bright": "#0F172A",
        "muted": "#64748B",
        "primary": "#6366F1",
        "primary_hover": "#4F46E5",
        "primary_light": "#EEF2FF",
        "accent": "#6366F1",
        "success": "#10B981",
        "success_bg": "#D1FAE5",
        "danger": "#EF4444",
        "danger_bg": "#FEE2E2",
        "danger_btn": "#EF4444",
        "warning": "#F59E0B",
        "card_bg": "#FFFFFF",
        "pill_bg": "#F1F5F9",
        "input_bg": "#FFFFFF",
        "scroll_trough": "#F8FAFC",
        "scroll_thumb": "#CBD5E1",
    }
}


# --- REUSABLE SCROLLABLE FRAME ---
class ScrollableFrame(tk.Frame):
    """Smooth scrollable container for long forms and dynamic card lists."""
    def __init__(self, parent, bg: str = "#181D2A", **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_window = tk.Frame(self.canvas, bg=bg)

        self.scrollable_window.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_window, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.bind_mousewheel(self)
        self.bind_mousewheel(self.canvas)
        self.bind_mousewheel(self.scrollable_window)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.window_id, width=event.width)

    def bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"), add="+")
        widget.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"), add="+")

    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_bg(self, bg_color: str):
        self.configure(bg=bg_color)
        self.canvas.configure(bg=bg_color)
        self.scrollable_window.configure(bg=bg_color)


class SelectField(tk.Frame):
    """Chooser that opens a plain list window.

    A native menu or ttk popdown takes a mouse grab. On macOS that grab stays
    after the list closes, and every later click is lost. This list does not
    take a grab.
    """
    _popup_window = None
    _popup_owner = None

    def __init__(
        self,
        parent,
        textvariable: tk.StringVar,
        values=(),
        state: str = "readonly",
        width: int = 22,
        bg: str = "#0F1219",
        fg: str = "#F1F5F9",
        border: str = "#283147",
        accent: str = "#6366F1",
        muted: str = "#94A3B8",
        font=("SF Pro Text", 11),
        labels: Optional[Dict[str, str]] = None,
    ):
        super().__init__(parent, bg=bg, highlightthickness=1, highlightbackground=border, highlightcolor=accent, bd=0)
        self._var = textvariable
        self._values = [str(v) for v in values]
        self._labels = {str(k): str(v) for k, v in (labels or {}).items()}
        self._state = state
        self._bg = bg
        self._fg = fg
        self._border = border
        self._accent = accent
        self._muted = muted
        self._font = font
        self._width = width
        self._entry: Optional[tk.Entry] = None
        self._label: Optional[tk.Label] = None
        self._arrow: Optional[tk.Label] = None
        self._selected_handlers = []

        if state == "normal":
            self._entry = tk.Entry(
                self,
                textvariable=textvariable,
                width=width,
                bg=bg,
                fg=fg,
                insertbackground=fg,
                disabledbackground=bg,
                disabledforeground=fg,
                font=font,
                bd=0,
                highlightthickness=0,
                relief="flat",
            )
            self._entry.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=6)
        else:
            self._label = tk.Label(
                self,
                text=self._caption(textvariable.get()),
                width=width,
                anchor="w",
                bg=bg,
                fg=fg,
                font=font,
                padx=10,
                pady=6,
            )
            self._label.pack(side="left", fill="both", expand=True)
            self._label.bind("<Button-1>", self._popup)
            self._trace_id = textvariable.trace_add("write", self._sync_label)
            self.bind("<Destroy>", self._drop_trace, add="+")

        self._arrow = tk.Label(self, text="", bg=bg, fg=muted, font=font, padx=8, pady=6)
        self._arrow.pack(side="right", fill="y")
        self._arrow.bind("<Button-1>", self._popup)
        self.bind("<Button-1>", self._popup)
        self.bind("<Enter>", self._on_field_enter, add="+")
        self.bind("<Leave>", self._on_field_leave, add="+")

    def _drop_trace(self, event):
        if event.widget is not self:
            return
        trace_id = getattr(self, "_trace_id", None)
        if not trace_id:
            return
        try:
            self._var.trace_remove("write", trace_id)
        except Exception:
            pass

    def _sync_label(self, *_args):
        if self._label is None:
            return
        try:
            shown = self._caption(self._var.get())
            self._label.configure(text=shown, fg=self._fg if self._var.get() else self._muted)
        except tk.TclError:
            pass

    def _on_field_enter(self, event):
        if getattr(event, "widget", None) is not self:
            return
        try:
            self.configure(highlightbackground=self._accent)
        except tk.TclError:
            pass

    def _on_field_leave(self, event):
        try:
            if 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
                return
        except tk.TclError:
            return
        try:
            self.configure(highlightbackground=self._border)
        except tk.TclError:
            pass

    def __setitem__(self, key, value):
        if key == "values":
            self._values = [str(v) for v in value]
            return
        raise tk.TclError(f"unknown option {key}")

    def __getitem__(self, key):
        if key == "values":
            return tuple(self._values)
        raise tk.TclError(f"unknown option {key}")

    def bind(self, sequence=None, func=None, add=None):
        if sequence == "<<ComboboxSelected>>" and func is not None:
            self._selected_handlers.append(func)
            return None
        return super().bind(sequence, func, add)

    def _popup(self, _event=None):
        if not self._values:
            return "break"
        if SelectField._popup_owner is self and SelectField._popup_window is not None:
            self._close_popup()
            return "break"
        self._close_popup()
        # A second window (menu, help panel, overrideredirect toplevel) takes
        # the mouse on macOS and later clicks never arrive. This list is a
        # frame in the same window, so it cannot grab or deactivate it.
        self.update_idletasks()
        host = self.winfo_toplevel()
        # Rows are plain labels. A listbox paints its selection with the
        # system color and then draws the label in that same color, so the
        # current value disappears.
        top = tk.Frame(host, bg=self._border, padx=1, pady=1, bd=0)
        visible = min(8, len(self._values))
        canvas = tk.Canvas(top, bg=self._bg, highlightthickness=0, bd=0, height=visible * 32)
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=self._bg)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        current = self._var.get()
        for value in self._values:
            selected = value == current
            row = tk.Label(
                inner,
                text=self._caption(value),
                anchor="w",
                bg=self._accent if selected else self._bg,
                fg="#FFFFFF" if selected else self._fg,
                font=self._font,
                padx=10,
                pady=6,
            )
            row.pack(fill="x")
            row.bind("<Button-1>", lambda e, v=value: self._choose(v))
            row.bind("<Enter>", lambda e, r=row, v=value: self._paint_row(r, v, True))
            row.bind("<Leave>", lambda e, r=row, v=value: self._paint_row(r, v, False))
        if len(self._values) > visible:
            bar = ttk.Scrollbar(top, command=canvas.yview)
            bar.pack(side="right", fill="y")
            canvas.configure(yscrollcommand=bar.set)

        def _fit(_event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
                canvas.itemconfigure(window_id, width=canvas.winfo_width())
            except tk.TclError:
                pass

        inner.bind("<Configure>", _fit)
        canvas.bind("<Configure>", _fit)
        inner.update_idletasks()
        full_h = max(inner.winfo_reqheight(), 1)
        row_h = max(full_h // max(len(self._values), 1), 1)
        visible_h = row_h * visible
        canvas.configure(height=visible_h)
        width = max(self.winfo_width(), 220)
        height = visible_h + 2
        x = self.winfo_rootx() - host.winfo_rootx()
        y = self.winfo_rooty() - host.winfo_rooty() + self.winfo_height() + 2
        if y + height > host.winfo_height():
            y = max(0, self.winfo_rooty() - host.winfo_rooty() - height - 2)
        top.place(x=x, y=y, width=width, height=height)
        top.lift()
        SelectField._popup_window = top
        SelectField._popup_owner = self
        self._release_leftover_grabs()
        return "break"

    def _caption(self, value: str) -> str:
        if not value:
            return "Select..."
        return self._labels.get(str(value), str(value))

    def _paint_row(self, row, value, inside):
        selected = value == self._var.get()
        if selected:
            bg, fg = self._accent, "#FFFFFF"
        elif inside:
            bg, fg = self._border, self._fg
        else:
            bg, fg = self._bg, self._fg
        try:
            row.configure(bg=bg, fg=fg)
        except tk.TclError:
            pass

    def _choose(self, value):
        self._close_popup()
        if value is None:
            return "break"
        self._var.set(value)
        for handler in list(self._selected_handlers):
            try:
                handler(None)
            except Exception as exc:
                print(f"[SelectField Error]: {exc}", file=sys.stderr)
        return "break"

    @classmethod
    def _close_popup(cls):
        top = cls._popup_window
        cls._popup_window = None
        cls._popup_owner = None
        if top is not None:
            try:
                top.destroy()
            except tk.TclError:
                pass

    @classmethod
    def close_if_outside(cls, widget):
        top = cls._popup_window
        owner = cls._popup_owner
        if top is None:
            return
        current = widget
        while current is not None:
            if current is top or current is owner:
                return
            try:
                current = current.master
            except Exception:
                break
        cls._close_popup()

    def _release_leftover_grabs(self):
        try:
            holders = self.tk.call("grab", "current")
        except tk.TclError:
            return
        if not holders:
            return
        if isinstance(holders, str):
            holders = (holders,)
        for name in holders:
            viewable = 0
            try:
                if int(self.tk.call("winfo", "exists", name)):
                    viewable = int(self.tk.call("winfo", "viewable", name))
            except tk.TclError:
                viewable = 0
            if not viewable:
                try:
                    self.tk.call("grab", "release", name)
                except tk.TclError:
                    pass


# --- NATIVE INTERACTIVE ACTION BUTTON ---
# Kept alive for the life of the process. PyObjC does not retain the callable
# behind class_addMethod, and a collected IMP makes later clicks crash or vanish.
_TK_FIRST_MOUSE_SELECTOR = None
_MOUSE_EVENT_CALLBACK = None
_ORIG_MOUSE_IMP = None


def _tk_accepts_first_mouse(_view, _event):
    """Deliver the click that also activates the window."""
    return True


def enable_macos_window_activation():
    """
    Python launched outside Terminal is often an accessory process. macOS then
    refuses to make the Tk window key, and every click is swallowed as a failed
    activation. A regular activation policy lets the window receive clicks.

    Tk 8.6's TKContentView leaves NSView.acceptsFirstMouse as NO. On macOS the
    click that activates an inactive window is discarded, so after a dropdown,
    a file dialog, or any focus change the next click does nothing. Returning
    YES makes that click reach the widget.
    Must run after tk.Tk(): importing AppKit first replaces Tk's NSApplication
    and Tk aborts on an unrecognized macOSVersion selector.
    """
    if sys.platform != "darwin":
        return
    global _TK_FIRST_MOUSE_SELECTOR
    try:
        import objc
        from AppKit import NSApplication
        app = NSApplication.sharedApplication()
        # NSApplicationActivationPolicyRegular == 0. Without this, a Python
        # process started from an IDE never becomes the key app, and Cocoa
        # discards every mouse click aimed at a Tk widget.
        app.setActivationPolicy_(0)
        if _TK_FIRST_MOUSE_SELECTOR is None:
            _TK_FIRST_MOUSE_SELECTOR = objc.selector(
                _tk_accepts_first_mouse, signature=b"B@:@"
            )
        for window in list(app.windows()):
            view = window.contentView()
            if view is None:
                continue
            try:
                objc.classAddMethod(
                    view.class__(), b"acceptsFirstMouse:", _TK_FIRST_MOUSE_SELECTOR
                )
            except Exception:
                pass
        _install_resize_click_fix(app)
    except Exception:
        pass


def _queue_click_at(root_x: int, root_y: int):
    """Run the control under a screen point on the next Tk turn.

    Used when Tk itself is about to discard the mouse-down because the
    process is not the active app yet. The coordinates come from the event,
    not the live cursor.
    """
    root = tk._default_root
    if root is None:
        return

    def run():
        try:
            widget = root.winfo_containing(root_x, root_y)
        except tk.TclError:
            return
        SelectField.close_if_outside(widget)
        _activate_control(_control_from_widget(widget))

    try:
        root.after(0, run)
    except tk.TclError:
        pass


def _install_resize_click_fix(app):
    """Tk 8.6.12 sets mouseInResizeArea on any mouse enter, then discards
    every left click until the pointer leaves the window. Hover still
    arrives, so the controls look fine and do nothing.
    """
    global _MOUSE_EVENT_CALLBACK, _ORIG_MOUSE_IMP
    if _MOUSE_EVENT_CALLBACK is not None:
        return
    import ctypes
    import objc

    libobjc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    libobjc.object_getClass.restype = ctypes.c_void_p
    libobjc.object_getClass.argtypes = [ctypes.c_void_p]
    libobjc.sel_registerName.restype = ctypes.c_void_p
    libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
    libobjc.class_getInstanceMethod.restype = ctypes.c_void_p
    libobjc.class_getInstanceMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libobjc.method_setImplementation.restype = ctypes.c_void_p
    libobjc.method_setImplementation.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    cls_ptr = libobjc.object_getClass(objc.pyobjc_id(app))
    selector = libobjc.sel_registerName(b"tkProcessMouseEvent:")
    method = libobjc.class_getInstanceMethod(cls_ptr, selector)
    if not method:
        return
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )

    def _py_imp(self_ptr, sel_ptr, event_ptr):
        try:
            event = objc.objc_object(c_void_p=event_ptr)
            if event is not None and int(event.type()) == 1:
                window = event.window()
                if window is not None and window.respondsToSelector_(b"setMouseInResizeArea:"):
                    window.setMouseInResizeArea_(False)
                # Tk drops the click when NSApp is not active yet. Deliver
                # that click from the event location. A later resend hits
                # invoke's short repeat guard.
                if not app.isActive() and window is not None:
                    loc = event.locationInWindow()
                    frame = window.frame()
                    screen = window.screen().frame()
                    screen_top = float(screen.origin.y) + float(screen.size.height)
                    root_x = int(round(float(frame.origin.x) + float(loc.x)))
                    root_y = int(round(screen_top - (float(frame.origin.y) + float(loc.y))))
                    _queue_click_at(root_x, root_y)
        except Exception:
            pass
        return callback_type(_ORIG_MOUSE_IMP)(self_ptr, sel_ptr, event_ptr)

    _MOUSE_EVENT_CALLBACK = callback_type(_py_imp)
    _ORIG_MOUSE_IMP = libobjc.method_setImplementation(
        method, ctypes.cast(_MOUSE_EVENT_CALLBACK, ctypes.c_void_p)
    )


_COMBOBOX_GRAB_PATCH = r"""
proc ttk::combobox::MapPopdown {w} {
    catch {[winfo parent $w] state pressed}
}
proc ttk::combobox::UnmapPopdown {w} {
    catch {[winfo parent $w] state !pressed}
    catch {ttk::releaseGrab $w}
    catch {grab release $w}
}
proc ttk::combobox::Unpost {cb} {
    if {[winfo exists $cb.popdown]} {
        destroy $cb.popdown
    }
    foreach w [grab current] {
        catch {grab release $w}
        catch {ttk::releaseGrab $w}
    }
    catch {focus [winfo toplevel $cb]}
}
bind ComboboxListbox <Map> {}
"""


def widget_at_screen_point(root: tk.Misc, cocoa_x: float, cocoa_y: float):
    """Return the Tk widget under a Cocoa screen point.

    Cocoa's origin is the bottom-left of the display. Tk 8.6.12 on current
    macOS feeds the wrong Y into its own hit test, so clicks land above the
    pointer. Bottom-of-window controls then never see the click, and controls
    higher up only see it when the pointer is in the lower part of the widget.
    This conversion uses the window frame, which matches Tk's winfo_rootx/y.
    """
    if sys.platform != "darwin":
        return None
    try:
        from AppKit import NSApplication
    except Exception:
        return None
    try:
        windows = list(NSApplication.sharedApplication().windows())
    except Exception:
        return None
    for window in windows:
        try:
            frame = window.frame()
            origin_x = float(frame.origin.x)
            origin_y = float(frame.origin.y)
            width = float(frame.size.width)
            height = float(frame.size.height)
        except Exception:
            continue
        if not (origin_x <= cocoa_x < origin_x + width and origin_y <= cocoa_y < origin_y + height):
            continue
        try:
            local = window.convertPointFromScreen_((cocoa_x, cocoa_y))
            screen = window.screen().frame()
            screen_top = float(screen.origin.y) + float(screen.size.height)
            abs_x = int(round(origin_x + float(local.x)))
            abs_y = int(round(screen_top - (origin_y + float(local.y))))
            return root.winfo_containing(abs_x, abs_y)
        except Exception:
            return None
    return None


def widget_under_pointer(root: tk.Misc):
    """Widget under the real cursor, using Cocoa coordinates."""
    if sys.platform != "darwin":
        return None
    try:
        from AppKit import NSEvent
        loc = NSEvent.mouseLocation()
        return widget_at_screen_point(root, float(loc.x), float(loc.y))
    except Exception:
        return None


def _control_from_widget(widget):
    """Button, menu, or clickable card that should handle this leaf widget."""
    current = widget
    while current is not None:
        if isinstance(current, ActionButton):
            return None if current._disabled else current
        if isinstance(current, SelectField):
            if current._entry is not None and widget is current._entry:
                return None
            return current
        if getattr(current, "_click_command", None) is not None:
            return current
        try:
            current = current.master
        except Exception:
            return None
    return None


def _activate_control(control):
    if control is None:
        return
    try:
        if not control.winfo_exists():
            return
    except tk.TclError:
        return
    if isinstance(control, ActionButton):
        control.invoke()
        return
    if isinstance(control, SelectField):
        control._popup()
        return
    command = getattr(control, "_click_command", None)
    if command is not None:
        try:
            command()
        except Exception as exc:
            print(f"[Click Error]: {exc}", file=sys.stderr)


class ActionButton(tk.Label):
    """
    Click target drawn as a flat label.

    ttk.Button on macOS Cocoa arms on <Button-1> and fires only if the widget
    is still in the 'pressed' state on <ButtonRelease-1>. A cursor change, a
    style redraw, or a combobox grab synthesizes <Button1-Leave>, which clears
    'pressed' and drops the click. After that the button looks fine and does
    nothing. This label records the press itself and runs the command on the
    mouse-up wherever it lands, so a stray Leave or a retargeted release
    cannot cancel it.
    Colors change without touching padding or border, so hover cannot resize
    the widget and retrigger Enter/Leave.
    The command runs on mouse-down. Hover already shows Tk delivered the
    pointer to this label, so the click uses that same delivery instead of a
    second hit test. Waiting for mouse-up drops the click when the release
    is sent somewhere else.
    """

    def __init__(
        self,
        parent,
        text: str = "",
        command=None,
        bg: str = "#181D2A",
        fg: str = "#FFFFFF",
        hover_bg: Optional[str] = None,
        active_bg: Optional[str] = None,
        font=("SF Pro Text", 10, "bold"),
        padx: int = 12,
        pady: int = 6,
        image=None,
        compound: str = "left",
        cursor: str = "pointinghand",
        anchor: str = "center",
        style: Optional[str] = None,
        **kwargs
    ):
        self._custom_command = command
        self._normal_bg = bg
        self._normal_fg = fg
        self._hover_bg = hover_bg or bg
        self._active_bg = active_bg or hover_bg or bg
        self._disabled = False
        self._pressed = False
        self._pointer_inside = False
        self._image_ref = image

        label_options = dict(
            text=text,
            bg=bg,
            fg=fg,
            font=font,
            padx=padx,
            pady=pady,
            compound=compound,
            cursor=cursor,
            anchor=anchor,
            bd=0,
            relief="flat",
            highlightthickness=0,
            takefocus=False,
        )
        if image is not None:
            label_options["image"] = image
        super().__init__(parent, **label_options)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release_visual)

    def _paint(self, bg: str):
        fg = "#94A3B8" if self._disabled else self._normal_fg
        try:
            super().configure(bg=bg, fg=fg)
        except tk.TclError:
            pass

    def _on_enter(self, _event):
        if self._pointer_inside:
            return
        self._pointer_inside = True
        if not self._pressed and not self._disabled:
            self._paint(self._hover_bg)

    def _on_leave(self, event):
        # Cursor changes on macOS echo a Leave whose coordinates are still
        # inside the widget. Repainting there starts an Enter/Leave storm.
        try:
            if 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
                return
        except tk.TclError:
            return
        self._pointer_inside = False
        if not self._pressed:
            self._paint(self._normal_bg)

    def _on_press(self, _event):
        if self._disabled:
            return
        self._pressed = True
        self._paint(self._active_bg)
        self.invoke()

    def _on_release_visual(self, _event):
        self._pressed = False
        self._paint(self._hover_bg if self._pointer_inside and not self._disabled else self._normal_bg)

    def invoke(self):
        if self._disabled or not self._custom_command:
            return
        now = time.monotonic()
        if now - getattr(self, "_last_invoke_at", 0.0) < 0.3:
            return
        self._last_invoke_at = now
        try:
            self._custom_command()
        except Exception as e:
            print(f"[ActionButton Error]: {e}", file=sys.stderr)

    def update_style(
        self,
        bg: Optional[str] = None,
        fg: Optional[str] = None,
        hover_bg: Optional[str] = None,
        active_bg: Optional[str] = None,
        text: Optional[str] = None,
        image=None,
    ):
        if bg is not None:
            self._normal_bg = bg
        if fg is not None:
            self._normal_fg = fg
        if hover_bg is not None:
            self._hover_bg = hover_bg
        if active_bg is not None:
            self._active_bg = active_bg
        options = {}
        if text is not None:
            options["text"] = text
        if image is not None:
            self._image_ref = image
            options["image"] = image
        if options:
            try:
                super().configure(**options)
            except tk.TclError:
                pass
        if self._pressed:
            shown = self._active_bg
        elif self._pointer_inside and not self._disabled:
            shown = self._hover_bg
        else:
            shown = self._normal_bg
        self._paint(shown)

    def configure(self, cnf=None, **kwargs):
        if cnf and not isinstance(cnf, dict):
            return super().configure(cnf)
        if isinstance(cnf, dict):
            kwargs = {**cnf, **kwargs}
        kwargs.pop("style", None)
        style_keys = {}
        for key in ("bg", "fg", "hover_bg", "active_bg"):
            if key in kwargs:
                style_keys[key] = kwargs.pop(key)
        if "command" in kwargs:
            self._custom_command = kwargs.pop("command")
        state_changed = "state" in kwargs
        if state_changed:
            self._disabled = kwargs.pop("state") in ("disabled", tk.DISABLED)
        if "image" in kwargs and kwargs["image"] is not None:
            self._image_ref = kwargs["image"]
        if style_keys:
            self.update_style(**style_keys)
        elif state_changed:
            shown = self._hover_bg if self._pointer_inside and not self._disabled else self._normal_bg
            self._paint(shown)
        if kwargs:
            return super().configure(**kwargs)
        return None

    config = configure

    def cget(self, key):
        if key in ("bg", "background"):
            return self._normal_bg
        if key in ("fg", "foreground"):
            return self._normal_fg
        if key == "command":
            return self._custom_command
        if key == "state":
            return "disabled" if self._disabled else "normal"
        return super().cget(key)

    def __getitem__(self, key):
        return self.cget(key)



# --- MAIN APPLICATION CONTROLLER & VIEW ---
class RealTimeAssistantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live Call Assistant -- Grounded ChatGPT Copilot")
        self.root.geometry("1200x740+60+40")
        self.root.minsize(1000, 640)

        # Make this process a normal foreground app before the window is shown,
        # so the first click is delivered instead of being eaten as activation.
        enable_macos_window_activation()
        self.root.bind("<Map>", self._on_root_map, add="+")
        self.root.after(50, self._on_root_map)
        self.root.after(300, self._on_root_map)

        # Styling & Theme State
        self.theme_name = "midnight"
        self.theme = THEMES[self.theme_name]
        self.active_page = "copilot"

        # Audio & STT State
        self.audio_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self._poll_ui_queue()
        self.is_listening = False
        self.stream = None
        self.streams = []
        self._has_meeting_stream = False
        self.processing_thread = None
        self.current_samplerate = 16000
        self.transcript_history: List[Tuple[str, str, str]] = []
        self.suggestion_history: List[Tuple[str, str, str]] = []
        self._stt_generation = 0
        self._history_lock = threading.Lock()
        self.transcripts: List[Dict[str, str]] = []
        self.whisper_model: Optional[WhisperModel] = None
        self._whisper_lock = threading.Lock()
        self._loaded_whisper_tag: Optional[str] = None
        self.mic_devices: List[Tuple[str, str]] = [("auto", "System default microphone")]
        self.meeting_devices: List[Tuple[str, str]] = [("auto", "BlackHole if available")]

        # AI & Storage Engine State
        self.chroma_client = None
        self.collection = None
        self.llm_client = None
        self.llm_model_name = "llama3.1:8b"
        self.retrieved_sources: List[str] = []
        self.discovered_models: List[str] = []
        self.review_packets_list: List[str] = []
        self.selected_packet_name: Optional[str] = None

        # Reactive Tk Variables
        self.selected_profile = tk.StringVar(value="Default_Client")
        self.selected_mic = tk.StringVar(value="auto")
        self.selected_meeting = tk.StringVar(value="auto")
        self.llm_provider = tk.StringVar(value="ollama")
        self.ollama_url_input = tk.StringVar(value=DEFAULT_CONFIG["ollama_url"])
        self.openai_key_input = tk.StringVar(value="")
        self.custom_url_input = tk.StringVar(value=DEFAULT_CONFIG["custom_url"])
        self.custom_key_input = tk.StringVar(value="")
        self.selected_model = tk.StringVar(value=DEFAULT_CONFIG["selected_model"])
        self.stt_language = tk.StringVar(value=DEFAULT_CONFIG["stt_language"])
        self.whisper_model_option = tk.StringVar(value=DEFAULT_CONFIG["whisper_model"])
        self.temperature = tk.StringVar(value=str(DEFAULT_CONFIG["temperature"]))
        self.max_tokens = tk.StringVar(value=str(DEFAULT_CONFIG["max_tokens"]))
        self.silence_threshold = tk.StringVar(value=str(DEFAULT_CONFIG["silence_threshold"]))
        self.system_prompt_text = DEFAULT_SYSTEM_PROMPT

        # Status & Header Indicators
        self.status_text = "SYSTEM READY"
        self.status_color = "#34D399"
        self.active_device_label = "Scanning audio devices..."
        self.hardware_sr_label = "16000 Hz (-> 16kHz)"
        self.engine_model_label = "Ollama (llama3.1:8b)"
        self.suggestion_text = (
            "Active Profile: [Default_Client]\n"
            "Knowledge Memory: 0 indexed context chunks.\n\n"
            "Start call session to view real-time grounded suggestions."
        )

        # Load persisted settings
        self.load_config()

        # Initialize Engines
        self.init_chromadb()
        self.init_llm_client()
        self.init_whisper_engine()

        # Build UI Architecture
        self.setup_ttk_styles()
        self._install_click_guard()
        self.build_ui()

        # Scan Devices & Profiles
        self.refresh_profiles()
        self.refresh_audio_devices()
        self.refresh_review_packets()

        # Protocol handler for clean window close
        self.root.protocol("WM_DELETE_WINDOW", self.initiate_shutdown)

    def safe_after(self, delay: int, callback, *args):
        """Safely dispatches callback to main Tkinter event loop from any background thread via queue."""
        if delay <= 0:
            self.ui_queue.put((callback, args))
        else:
            def delayed():
                time.sleep(delay / 1000.0)
                self.ui_queue.put((callback, args))
            threading.Thread(target=delayed, daemon=True).start()

    def _poll_ui_queue(self):
        """Processes queued UI callbacks on the main Tkinter thread."""
        try:
            while not self.ui_queue.empty():
                cb, args = self.ui_queue.get_nowait()
                cb(*args)
        except Exception:
            pass
        try:
            self.root.after(25, self._poll_ui_queue)
        except Exception:
            pass

    def _on_root_map(self, _event=None):
        """Raise the window once, after it is actually on screen."""
        if getattr(self, "_did_activate_window", False):
            return
        try:
            if not self.root.winfo_viewable():
                return
        except tk.TclError:
            return
        self._did_activate_window = True
        self.activate_macos_window()

    def activate_macos_window(self):
        """
        Bring the window forward once. Flipping -topmost while a click is in
        progress makes Cocoa drop that click, so this runs a single time on map
        and never calls update_idletasks (which re-enters the event loop).
        """
        enable_macos_window_activation()

        def async_activate():
            try:
                subprocess.run(
                    ["osascript", "-e", f'tell application "System Events" to set frontmost of first process whose unix id is {os.getpid()} to true'],
                    capture_output=True,
                    timeout=1.0,
                    check=False
                )
            except Exception:
                pass

        threading.Thread(target=async_activate, daemon=True).start()

        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.focus_force()
            self.root.after(300, self._release_topmost)
        except tk.TclError:
            pass

    def _release_topmost(self):
        try:
            self.root.attributes("-topmost", False)
            self.root.lift()
        except tk.TclError:
            pass

    def _patch_combobox_grabs(self):
        """Drop the global grab ttk installs while a dropdown is open.

        On macOS that grab survives after the popdown is withdrawn. Every later
        click is delivered to the invisible popdown, so buttons stop working.
        """
        try:
            self.root.tk.eval(_COMBOBOX_GRAB_PATCH)
        except tk.TclError:
            pass

    def _install_click_guard(self):
        self._patch_combobox_grabs()
        if getattr(self, "_click_guard_installed", False):
            return
        self._click_guard_installed = True
        self.root.bind_all("<ButtonPress-1>", self._on_global_button_press, add="+")
        self._schedule_grab_reaper()

    def _on_global_button_press(self, event):
        # The widget under the pointer already ran its own click. This only
        # drops a grab left behind by a menu, which would otherwise swallow
        # every later click, and closes an open chooser list.
        self._release_dead_grabs()
        SelectField.close_if_outside(getattr(event, "widget", None))

    def _release_dead_grabs(self):
        try:
            holders = self.root.tk.call("grab", "current")
        except tk.TclError:
            return
        if not holders:
            return
        if isinstance(holders, str):
            holders = (holders,)
        for name in holders:
            viewable = 0
            kind = ""
            try:
                if int(self.root.tk.call("winfo", "exists", name)):
                    viewable = int(self.root.tk.call("winfo", "viewable", name))
                    kind = str(self.root.tk.call("winfo", "class", name))
            except tk.TclError:
                viewable = 0
            # A menu grab stays "viewable" after the menu is gone and then
            # eats every click. Dialog grabs are left alone while they show.
            if (not viewable) or kind in ("Menu", "ComboboxPopdown"):
                try:
                    self.root.tk.call("grab", "release", name)
                except tk.TclError:
                    pass

    def _schedule_grab_reaper(self):
        self._release_dead_grabs()
        try:
            self.root.after(400, self._schedule_grab_reaper)
        except tk.TclError:
            pass

    # --- THEME & STYLES ---
    def setup_ttk_styles(self):
        """Configures clean clam TTK dropdown and scrollbar styles matching active theme."""
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        t = self.theme

        # Style TTK Combobox cleanly without any global grab interference
        style.configure("TCombobox",
            fieldbackground=t["input_bg"],
            background=t["surface_alt"],
            foreground=t["text_bright"],
            arrowcolor=t["text"],
            darkcolor=t["border"],
            lightcolor=t["border"],
            bordercolor=t["border"],
            padding=5
        )
        style.map("TCombobox",
            fieldbackground=[("readonly", t["input_bg"])],
            selectbackground=[("readonly", t["input_bg"])],
            selectforeground=[("readonly", t["text_bright"])]
        )

        # Slim thumb, no arrow buttons. arrowsize is the bar thickness in the clam theme.
        for axis, sticky in (("Vertical", "ns"), ("Horizontal", "we")):
            style.layout(f"{axis}.TScrollbar", [
                (f"{axis}.Scrollbar.trough", {
                    "sticky": sticky,
                    "children": [
                        (f"{axis}.Scrollbar.thumb", {"sticky": "nswe", "expand": True}),
                    ],
                })
            ])
        style.configure("TScrollbar",
            troughcolor=t["scroll_trough"],
            background=t["scroll_thumb"],
            darkcolor=t["scroll_thumb"],
            lightcolor=t["scroll_thumb"],
            bordercolor=t["scroll_trough"],
            arrowcolor=t["scroll_trough"],
            gripcount=0,
            arrowsize=8,
            relief="flat",
            borderwidth=0,
        )
        style.map("TScrollbar", background=[("active", t["muted"]), ("pressed", t["muted"])])
        self._patch_combobox_grabs()

    def toggle_theme(self):
        """Switches between Midnight and Daylight themes seamlessly."""
        self.theme_name = "daylight" if self.theme_name == "midnight" else "midnight"
        self.theme = THEMES[self.theme_name]
        self.setup_ttk_styles()

        # Update root & structural backgrounds
        self.root.configure(bg=self.theme["bg"])
        self.sidebar_frame.configure(bg=self.theme["surface"])
        self.right_container.configure(bg=self.theme["bg"])
        self.header_frame.configure(bg=self.theme["surface"])
        self.content_area.configure(bg=self.theme["bg"])

        if hasattr(self, "hdr_left") and self.hdr_left.winfo_exists():
            self.hdr_left.configure(bg=self.theme["surface"])
        if hasattr(self, "hdr_right") and self.hdr_right.winfo_exists():
            self.hdr_right.configure(bg=self.theme["surface"])
        if hasattr(self, "hdr_center") and self.hdr_center.winfo_exists():
            self.hdr_center.configure(bg=self.theme["surface"])

        # Update nav buttons & sidebar items
        self.update_sidebar_theme()
        self.update_header()
        self.show_page(self.active_page)

    # --- UI LAYOUT & NAVIGATION DRAWER ---
    def build_ui(self):
        """Constructs sidebar, header, and multi-page viewport."""
        self.root.configure(bg=self.theme["bg"])

        # Main horizontal split: Sidebar on Left, Content on Right
        self.sidebar_frame = tk.Frame(self.root, bg=self.theme["surface"], width=230, bd=0)
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)

        self.right_container = tk.Frame(self.root, bg=self.theme["bg"], bd=0)
        self.right_container.pack(side="right", fill="both", expand=True)

        # Header Strip at Top of Right Container
        self.header_frame = tk.Frame(self.right_container, bg=self.theme["surface"], height=72, bd=0, padx=24, pady=10)
        self.header_frame.pack(side="top", fill="x")
        self.header_frame.pack_propagate(False)

        # Main Page Viewport below Header
        self.content_area = tk.Frame(self.right_container, bg=self.theme["bg"], bd=0, padx=20, pady=16)
        self.content_area.pack(side="top", fill="both", expand=True)

        self.build_sidebar()
        self.build_header()
        self.show_page("copilot")

    def build_sidebar(self):
        """Constructs left navigation drawer with branding, tabs, and action controls."""
        t = self.theme
        # Brand Emblem & Title
        brand_box = tk.Frame(self.sidebar_frame, bg=t["surface"], padx=16, pady=12)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)
        self.sidebar_frame.grid_rowconfigure(2, weight=1)
        brand_box.grid(row=0, column=0, sticky="ew")

        logo_img = get_tk_icon("logo", size=36)
        self.lbl_logo = tk.Label(brand_box, image=logo_img, bg=t["surface"])
        self.lbl_logo.image = logo_img
        self.lbl_logo.pack(anchor="w")

        self.lbl_brand_title = tk.Label(brand_box, text="Context Assistant", font=("SF Pro Display", 15, "bold"), fg=t["text_bright"], bg=t["surface"])
        self.lbl_brand_title.pack(anchor="w", pady=(4, 0))

        self.lbl_brand_sub = tk.Label(brand_box, text="Live Call Grounded Copilot", font=("SF Pro Text", 10), fg=t["muted"], bg=t["surface"])
        self.lbl_brand_sub.pack(anchor="w")

        # Navigation Buttons Shelf
        nav_box = tk.Frame(self.sidebar_frame, bg=t["surface"], padx=10, pady=6)
        nav_box.grid(row=1, column=0, sticky="ew")

        self.nav_btns = {}
        pages = [
            ("copilot", "Live Copilot", "mic"),
            ("profiles", "Knowledge & Profiles", "layers"),
            ("packets", "Review Packets", "file_text"),
            ("settings", "Settings & Models", "settings"),
        ]

        for key, label, icon_name in pages:
            is_active = (key == self.active_page)
            icon_col = t["primary"] if is_active else t["muted"]
            icon = get_tk_icon(icon_name, color=icon_col, size=16)
            btn = ActionButton(
                nav_box,
                text="  " + label,
                image=icon,
                compound="left",
                bg=t["surface_alt"] if is_active else t["surface"],
                fg=t["text_bright"] if is_active else t["muted"],
                hover_bg=t["surface_alt"],
                active_bg=t["border"],
                font=("SF Pro Text", 10, "bold" if is_active else "normal"),
                padx=12,
                pady=7,
                anchor="w",
                command=lambda p=key: self.show_page(p),
                cursor="pointinghand"
            )
            btn.pack(fill="x", pady=2)
            self.nav_btns[key] = (btn, icon_name)

        # Bottom drawer sits in its own row so the spacer cannot cover it.
        # A packed expander was stacking above these controls, and clicks on
        # Theme and Close landed in the empty spacer.
        bottom_box = tk.Frame(self.sidebar_frame, bg=t["surface"], padx=10, pady=10)
        # Keep these controls off the window's bottom resize edge. macOS
        # delivers hover there but swallows the click.
        bottom_box.grid(row=3, column=0, sticky="ew", pady=(0, 36))

        self.lbl_active_prof_pill = tk.Label(
            bottom_box,
            text=f"Profile: {self.selected_profile.get()}",
            font=("SF Pro Text", 9, "bold"),
            fg=t["muted"],
            bg=t["surface_alt"],
            padx=10,
            pady=4,
            relief="flat"
        )
        self.lbl_active_prof_pill.pack(fill="x", pady=(0, 6))

        theme_icon = get_tk_icon("sun" if self.theme_name == "midnight" else "moon", color=t["muted"], size=16)
        self.btn_theme = ActionButton(
            bottom_box,
            text="  Daylight Theme" if self.theme_name == "midnight" else "  Midnight Theme",
            image=theme_icon,
            compound="left",
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 9, "bold"),
            padx=10,
            pady=7,
            anchor="w",
            command=self.toggle_theme,
            cursor="pointinghand"
        )
        self.btn_theme.pack(fill="x", pady=(0, 4))

        close_icon = get_tk_icon("power", color="#F87171", size=16)
        self.btn_close = ActionButton(
            bottom_box,
            text="  Close Application",
            image=close_icon,
            compound="left",
            bg=t["danger_bg"],
            fg=t["danger"],
            hover_bg="#7F1D1D" if self.theme_name == "midnight" else "#FECACA",
            active_bg="#991B1B" if self.theme_name == "midnight" else "#FCA5A5",
            font=("SF Pro Text", 9, "bold"),
            padx=10,
            pady=7,
            anchor="w",
            command=self.initiate_shutdown,
            cursor="pointinghand"
        )
        self.btn_close.pack(fill="x")

        spacer = tk.Frame(self.sidebar_frame, bg=t["surface"])
        spacer.grid(row=2, column=0, sticky="nsew")

    def update_sidebar_theme(self):
        """Updates styling and theme attributes across sidebar controls."""
        t = self.theme
        self.sidebar_frame.configure(bg=t["surface"])
        for child in self.sidebar_frame.winfo_children():
            self._recursive_bg(child, t["surface"])

        logo_img = get_tk_icon("logo", size=36)
        self.lbl_logo.configure(image=logo_img, bg=t["surface"])
        self.lbl_logo.image = logo_img

        if hasattr(self, "lbl_brand_title") and self.lbl_brand_title.winfo_exists():
            self.lbl_brand_title.configure(fg=t["text_bright"], bg=t["surface"])
        if hasattr(self, "lbl_brand_sub") and self.lbl_brand_sub.winfo_exists():
            self.lbl_brand_sub.configure(fg=t["muted"], bg=t["surface"])

        self.lbl_active_prof_pill.configure(
            bg=t["surface_alt"],
            fg=t["muted"],
            text=f"Profile: {self.selected_profile.get()}"
        )

        theme_icon = get_tk_icon("sun" if self.theme_name == "midnight" else "moon", color=t["muted"], size=16)
        self.btn_theme.update_style(
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            text="  Daylight Theme" if self.theme_name == "midnight" else "  Midnight Theme",
            image=theme_icon
        )

        close_icon = get_tk_icon("power", color="#F87171", size=16)
        self.btn_close.update_style(
            bg=t["danger_bg"],
            fg=t["danger"],
            hover_bg="#7F1D1D" if self.theme_name == "midnight" else "#FECACA",
            active_bg="#991B1B" if self.theme_name == "midnight" else "#FCA5A5",
            text="  Close Application",
            image=close_icon
        )

        for key, (btn, icon_name) in self.nav_btns.items():
            is_active = (key == self.active_page)
            icon_col = t["primary"] if is_active else t["muted"]
            icon = get_tk_icon(icon_name, color=icon_col, size=16)
            btn.update_style(
                bg=t["surface_alt"] if is_active else t["surface"],
                fg=t["text_bright"] if is_active else t["muted"],
                hover_bg=t["surface_alt"],
                active_bg=t["border"],
                image=icon
            )

    def _recursive_bg(self, widget, bg_color):
        if isinstance(widget, ActionButton):
            return
        try:
            widget.configure(bg=bg_color)
        except Exception:
            pass
        for ch in widget.winfo_children():
            self._recursive_bg(ch, bg_color)

    def build_header(self):
        """Constructs 3-column top header with page info, centered status pill, and session controls."""
        t = self.theme
        # Left: Title and Subtitle
        self.header_frame.grid_columnconfigure(1, weight=1)
        self.header_frame.grid_rowconfigure(0, weight=1)
        self.hdr_left = tk.Frame(self.header_frame, bg=t["surface"])
        self.hdr_left.grid(row=0, column=0, sticky="nsw")

        self.hdr_title = tk.Label(self.hdr_left, text="Live Call Copilot", font=("SF Pro Display", 16, "bold"), fg=t["text_bright"], bg=t["surface"])
        self.hdr_title.pack(anchor="w")

        self.hdr_sub = tk.Label(self.hdr_left, text="Real-time speech transcription & verified suggestions", font=("SF Pro Text", 10), fg=t["muted"], bg=t["surface"])
        self.hdr_sub.pack(anchor="w")

        # Right: Source Device Pill and Toggled Session Button
        self.hdr_right = tk.Frame(self.header_frame, bg=t["surface"])
        self.hdr_right.grid(row=0, column=2, sticky="nse")

        self.hdr_source_pill = tk.Label(
            self.hdr_right,
            text=f"Source: {self.active_device_label}",
            font=("SF Pro Text", 9),
            fg=t["muted"],
            bg=t["surface_alt"],
            padx=10,
            pady=4,
            relief="flat"
        )
        self.hdr_source_pill.pack(side="left", padx=(0, 10))

        play_icon = get_tk_icon("play", color="#0F172A", size=14)
        self.btn_session = ActionButton(
            self.hdr_right,
            text="  Start Session",
            image=play_icon,
            compound="left",
            bg=t["success"],
            fg="#0F172A",
            hover_bg="#10B981",
            active_bg="#059669",
            font=("SF Pro Text", 10, "bold"),
            padx=14,
            pady=7,
            command=self.toggle_session,
            cursor="pointinghand"
        )
        self.btn_session.pack(side="right")

        # Center: Live Status Indicator Pill (expands into middle cavity)
        self.hdr_center = tk.Frame(self.header_frame, bg=t["surface"])
        self.hdr_center.grid(row=0, column=1, sticky="nsew")

        dot_icon = get_tk_icon("dot", color=self.status_color, size=10)
        self.hdr_status_pill = tk.Label(
            self.hdr_center,
            text=f"  {self.status_text}",
            image=dot_icon,
            compound="left",
            font=("SF Pro Text", 10, "bold"),
            fg=self.status_color,
            bg=t["surface_alt"],
            padx=14,
            pady=4,
            relief="flat"
        )
        self.hdr_status_pill.image = dot_icon
        self.hdr_status_pill.pack(anchor="center", pady=6)

    def update_header(self):
        """Refreshes header status text, source device label, and session button state."""
        t = self.theme
        titles = {
            "copilot": ("Live Call Copilot", "Real-time speech transcription & verified suggestions"),
            "profiles": ("Knowledge & Profiles", "Manage isolated ChromaDB client vector collections"),
            "packets": ("Review Packets", "Post-call workspace memory sync records"),
            "settings": ("Settings & Models", "Configure LLM provider, Whisper STT, and prompt tuning"),
        }
        title, sub = titles.get(self.active_page, ("Copilot", ""))
        self.hdr_title.config(text=title, fg=t["text_bright"], bg=t["surface"])
        self.hdr_sub.config(text=sub, fg=t["muted"], bg=t["surface"])

        dot_icon = get_tk_icon("dot", color=self.status_color, size=10)
        self.hdr_status_pill.config(
            text=f"  {self.status_text}",
            image=dot_icon,
            fg=self.status_color,
            bg=t["surface_alt"]
        )
        self.hdr_status_pill.image = dot_icon

        short_dev = self.active_device_label
        if len(short_dev) > 42:
            short_dev = short_dev[:39] + "..."
        self.hdr_source_pill.config(
            text=f"Source: {short_dev}",
            fg=t["muted"],
            bg=t["surface_alt"]
        )

        if self.is_listening:
            stop_icon = get_tk_icon("square", color="#FFFFFF", size=14)
            self.btn_session.update_style(
                bg=t["danger_btn"],
                fg="#FFFFFF",
                hover_bg="#E11D48",
                active_bg="#BE123C",
                text="  Stop & Save",
                image=stop_icon
            )
        else:
            play_icon = get_tk_icon("play", color="#0F172A", size=14)
            self.btn_session.update_style(
                bg=t["success"],
                fg="#0F172A",
                hover_bg="#10B981",
                active_bg="#059669",
                text="  Start Session",
                image=play_icon
            )

    def show_page(self, page_name: str):
        """Switches active view frame between Copilot, Profiles, Packets, and Settings."""
        self.active_page = page_name

        # Update sidebar button styles and icon highlights
        for key, (btn, icon_name) in self.nav_btns.items():
            is_active = (key == page_name)
            icon_col = self.theme["primary"] if is_active else self.theme["muted"]
            icon = get_tk_icon(icon_name, color=icon_col, size=16)
            btn.update_style(
                bg=self.theme["surface_alt"] if is_active else self.theme["surface"],
                fg=self.theme["text_bright"] if is_active else self.theme["muted"],
                hover_bg=self.theme["surface_alt"],
                active_bg=self.theme["border"],
                image=icon
            )

        # Clear existing view
        for child in self.content_area.winfo_children():
            child.destroy()

        self.update_header()

        if page_name == "copilot":
            self.build_copilot_page()
        elif page_name == "profiles":
            self.build_profiles_page()
        elif page_name == "packets":
            self.build_packets_page()
        elif page_name == "settings":
            self.build_settings_page()

    # --- PAGE 1: LIVE COPILOT COMMAND CENTER ---
    def build_copilot_page(self):
        """Renders 4-card metrics strip and split dialogue / grounded suggestion console."""
        t = self.theme
        container = tk.Frame(self.content_area, bg=t["bg"])
        container.pack(fill="both", expand=True)

        # Top Metrics Strip
        metrics_frame = tk.Frame(container, bg=t["bg"])
        metrics_frame.pack(fill="x", pady=(0, 14))

        metrics = [
            ("WORKSPACE PROFILE", self.selected_profile.get(), t["primary"]),
            ("HARDWARE AUDIO", self.hardware_sr_label, t["success"]),
            ("INFERENCE ENGINE", self.engine_model_label, t["text"]),
            ("ZERO HALLUCINATION", "Active & Grounded", t["success"]),
        ]

        for idx, (label, val, col) in enumerate(metrics):
            card = tk.Frame(metrics_frame, bg=t["surface"], padx=14, pady=10, highlightbackground=t["border"], highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=(0 if idx == 0 else 8, 0))

            tk.Label(card, text=label, font=("SF Pro Text", 9, "bold"), fg=t["muted"], bg=t["surface"]).pack(anchor="w")
            tk.Label(card, text=val, font=("SF Pro Text", 12, "bold"), fg=col, bg=t["surface"]).pack(anchor="w", pady=(2, 0))

        # Main Split Row (50% / 50%)
        split_frame = tk.Frame(container, bg=t["bg"])
        split_frame.pack(fill="both", expand=True)

        # Left Panel: Live Call Dialogue Stream
        self.left_panel = tk.Frame(split_frame, bg=t["surface"], padx=16, pady=14, highlightbackground=t["border"], highlightthickness=1)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

        left_hdr = tk.Frame(self.left_panel, bg=t["surface"])
        left_hdr.pack(fill="x", pady=(0, 10))

        mic_icon = get_tk_icon("mic", color=t["text_bright"], size=16)
        lbl_mic_hdr = tk.Label(left_hdr, text="  Live Call Dialogue Stream", image=mic_icon, compound="left", font=("SF Pro Text", 13, "bold"), fg=t["text_bright"], bg=t["surface"])
        lbl_mic_hdr.image = mic_icon
        lbl_mic_hdr.pack(side="left")

        self.lbl_phrase_count = tk.Label(
            left_hdr,
            text=f"{len(self.transcripts)} phrases",
            font=("SF Pro Text", 9, "bold"),
            fg=t["primary"],
            bg=t["surface_alt"],
            padx=8,
            pady=2
        )
        self.lbl_phrase_count.pack(side="left", padx=8)

        btn_export = ActionButton(
            left_hdr,
            text="  Export",
            image=get_tk_icon("file_text", color=t["text"], size=14),
            compound="left",
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 9, "bold"),
            padx=10,
            pady=4,
            command=lambda: self.open_export_dialog("live"),
            cursor="pointinghand"
        )
        btn_export.pack(side="right", padx=(0, 6))

        btn_clear = ActionButton(
            left_hdr,
            text="  Clear",
            image=get_tk_icon("trash", color=t["text"], size=14),
            compound="left",
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 9, "bold"),
            padx=10,
            pady=4,
            command=self.clear_transcripts,
            cursor="pointinghand"
        )
        btn_clear.pack(side="right")

        self.transcripts_scroll = ScrollableFrame(self.left_panel, bg=t["surface"])
        self.transcripts_scroll.pack(fill="both", expand=True)

        # Right Panel: Grounded Suggestion Console
        self.right_panel = tk.Frame(split_frame, bg=t["surface"], padx=16, pady=14, highlightbackground=t["success"], highlightthickness=1.5)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=(8, 0))

        right_hdr = tk.Frame(self.right_panel, bg=t["surface"])
        right_hdr.pack(fill="x", pady=(0, 10))

        pill_sugg = tk.Label(
            right_hdr,
            text="GROUNDED SUGGESTION",
            font=("SF Pro Text", 9, "bold"),
            fg="#000000",
            bg=t["success"],
            padx=8,
            pady=3
        )
        pill_sugg.pack(side="left")

        brain_icon = get_tk_icon("brain", color=t["muted"], size=16)
        lbl_what = tk.Label(right_hdr, text="  What to say aloud:", image=brain_icon, compound="left", font=("SF Pro Text", 11, "bold"), fg=t["muted"], bg=t["surface"])
        lbl_what.image = brain_icon
        lbl_what.pack(side="left", padx=8)

        copy_icon = get_tk_icon("copy", color=t["text"], size=14)
        self.btn_copy_sugg = ActionButton(
            right_hdr,
            text="  Copy",
            image=copy_icon,
            compound="left",
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 9, "bold"),
            padx=12,
            pady=4,
            command=self.copy_suggestion_to_clipboard,
            cursor="pointinghand"
        )
        self.btn_copy_sugg.pack(side="right")

        self.suggestions_scroll = ScrollableFrame(self.right_panel, bg=t["surface"])
        self.suggestions_scroll.pack(fill="both", expand=True)

        self.render_transcripts_view()
        self.render_suggestions_view()

    def render_transcripts_view(self):
        """Populates dialogue cards descending with newest on top and LATEST badge."""
        t = self.theme
        win = self.transcripts_scroll.scrollable_window
        for ch in win.winfo_children():
            ch.destroy()

        self.lbl_phrase_count.config(text=f"{len(self.transcripts)} phrases")

        if not self.transcripts:
            empty_card = tk.Frame(win, bg=t["surface_alt"], padx=20, pady=36, highlightbackground=t["border"], highlightthickness=1)
            empty_card.pack(fill="x", pady=20)
            mic_big = get_tk_icon("mic", color=t["muted"], size=36)
            lbl_empty_mic = tk.Label(empty_card, image=mic_big, bg=t["surface_alt"])
            lbl_empty_mic.image = mic_big
            lbl_empty_mic.pack()
            tk.Label(empty_card, text="Audio Stream Monitoring", font=("SF Pro Text", 13, "bold"), fg=t["text_bright"], bg=t["surface_alt"]).pack(pady=(6, 2))
            tk.Label(
                empty_card,
                text="When live participants speak, conversation phrases are segmented\nwith adaptive energy VAD and transcribed here in descending order.",
                font=("SF Pro Text", 10),
                fg=t["muted"],
                bg=t["surface_alt"],
                justify="center"
            ).pack()
            return

        for idx, item in enumerate(self.transcripts):
            is_latest = (idx == 0)
            border_col = t["primary"] if is_latest else t["border"]
            card = tk.Frame(win, bg=t["surface_alt"], padx=12, pady=10, highlightbackground=border_col, highlightthickness=1.5 if is_latest else 1)
            card.pack(fill="x", pady=4)

            top_row = tk.Frame(card, bg=t["surface_alt"])
            top_row.pack(fill="x", pady=(0, 4))

            tk.Label(top_row, text=item["time"], font=("SF Pro Text", 8, "bold"), fg=t["muted"], bg=t["surface"], padx=6, pady=2).pack(side="left", padx=(0, 6))
            tk.Label(top_row, text=item["speaker"], font=("SF Pro Text", 8, "bold"), fg=t["primary"], bg=t["surface"], padx=6, pady=2).pack(side="left")

            if is_latest:
                tk.Label(top_row, text="LATEST", font=("SF Pro Text", 8, "bold"), fg="#FFFFFF", bg=t["primary"], padx=6, pady=2).pack(side="left", padx=6)

            txt_lbl = tk.Label(
                card,
                text=item["text"],
                font=("SF Pro Text", 11, "bold" if is_latest else "normal"),
                fg=t["text_bright"] if is_latest else t["text"],
                bg=t["surface_alt"],
                wraplength=480,
                justify="left",
                anchor="w"
            )
            txt_lbl.pack(fill="x", pady=(2, 0))

    def render_suggestions_view(self):
        """Renders grounded suggestion card along with ChromaDB Top-3 verified citations."""
        t = self.theme
        win = self.suggestions_scroll.scrollable_window
        for ch in win.winfo_children():
            ch.destroy()

        # Main Grounded Suggestion Text Card
        main_card = tk.Frame(win, bg=t["surface_alt"], padx=16, pady=16, highlightbackground=t["border"], highlightthickness=1)
        main_card.pack(fill="x", pady=(0, 12))

        sugg_lbl = tk.Label(
            main_card,
            text=self.suggestion_text,
            font=("SF Pro Text", 13, "bold"),
            fg=t["text_bright"],
            bg=t["surface_alt"],
            wraplength=480,
            justify="left",
            anchor="w"
        )
        sugg_lbl.pack(fill="x")

        # Verified Knowledge Citations (ChromaDB Top-3 Chunks)
        if self.retrieved_sources:
            sep = tk.Frame(win, bg=t["border"], height=1)
            sep.pack(fill="x", pady=10)

            hdr = tk.Label(
                win,
                text="VERIFIED KNOWLEDGE SOURCES (CHROMADB TOP-3):",
                font=("SF Pro Text", 9, "bold"),
                fg=t["muted"],
                bg=t["surface"]
            )
            hdr.pack(anchor="w", pady=(0, 6))

            for idx, src in enumerate(self.retrieved_sources):
                c_card = tk.Frame(win, bg=t["surface_alt"], padx=10, pady=8, highlightbackground=t["border"], highlightthickness=1)
                c_card.pack(fill="x", pady=4)

                top_c = tk.Frame(c_card, bg=t["surface_alt"])
                top_c.pack(fill="x", pady=(0, 2))

                tk.Label(top_c, text=f"Chunk #{idx+1}", font=("SF Pro Text", 8, "bold"), fg=t["primary"], bg=t["surface"], padx=6, pady=1).pack(side="left")

                tk.Label(
                    c_card,
                    text=src,
                    font=("SF Pro Text", 9),
                    fg=t["muted"],
                    bg=t["surface_alt"],
                    wraplength=480,
                    justify="left",
                    anchor="w"
                ).pack(fill="x")

    def clear_transcripts(self):
        """Clears active dialogue stream."""
        self.transcripts.clear()
        self.transcript_history.clear()
        self.suggestion_history.clear()
        self.render_transcripts_view()

    def copy_suggestion_to_clipboard(self):
        """Copies suggestion text to system clipboard via macOS pbcopy with feedback animation."""
        text = self.suggestion_text.strip()
        if not text:
            return

        try:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            check_icon = get_tk_icon("check", color=self.theme["success"], size=14)
            self.btn_copy_sugg.config(text="  Copied!", image=check_icon)
            self.btn_copy_sugg.image = check_icon

            def restore_copy():
                copy_icon = get_tk_icon("copy", color=self.theme["text"], size=14)
                self.btn_copy_sugg.config(text="  Copy", image=copy_icon)
                self.btn_copy_sugg.image = copy_icon

            self.root.after(2000, restore_copy)
        except Exception as e:
            messagebox.showerror("Clipboard Error", f"Could not copy text: {e}")

    # --- PAGE 2: CONTEXT & PROFILES MANAGER ---
    def build_profiles_page(self):
        """Renders profile switcher, creation, import context, and indexed memories inspection."""
        t = self.theme
        container = tk.Frame(self.content_area, bg=t["bg"])
        container.pack(fill="both", expand=True)

        # Header Control Card
        ctrl_card = tk.Frame(container, bg=t["surface"], padx=18, pady=14, highlightbackground=t["border"], highlightthickness=1)
        ctrl_card.pack(fill="x", pady=(0, 14))

        top_r = tk.Frame(ctrl_card, bg=t["surface"])
        top_r.pack(fill="x")

        tk.Label(top_r, text="Workspace Knowledge Profiles", font=("SF Pro Text", 14, "bold"), fg=t["text_bright"], bg=t["surface"]).pack(side="left")
        tk.Label(top_r, text="Each client maintains an isolated ChromaDB vector index.", font=("SF Pro Text", 10), fg=t["muted"], bg=t["surface"]).pack(side="left", padx=12)

        # Controls row
        btn_r = tk.Frame(ctrl_card, bg=t["surface"])
        btn_r.pack(fill="x", pady=(12, 0))

        tk.Label(btn_r, text="Select Profile:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"]).pack(side="left", padx=(0, 8))

        subdirs = self.get_profile_names()
        self.combo_profiles = SelectField(btn_r, textvariable=self.selected_profile, values=subdirs, state="readonly", width=22, bg=t["input_bg"], fg=t["text"], border=t["border"], accent=t["primary"], muted=t["muted"])
        self.combo_profiles.pack(side="left", padx=(0, 12))
        self.combo_profiles.bind("<<ComboboxSelected>>", self.on_profile_dropdown_changed)

        plus_icon = get_tk_icon("plus", color="#0F172A", size=14)
        btn_new_prof = ActionButton(
            btn_r,
            text="  New Profile",
            image=plus_icon,
            compound="left",
            bg=t["primary"],
            fg="#0F172A" if self.theme_name == "midnight" else "#FFFFFF",
            hover_bg=t["primary_hover"],
            active_bg=t["primary_light"],
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=6,
            command=self.create_new_profile_dialog,
            cursor="pointinghand"
        )
        btn_new_prof.pack(side="left", padx=(0, 6))

        up_icon = get_tk_icon("upload", color=t["text_bright"], size=14)
        btn_imp = ActionButton(
            btn_r,
            text="  Import Context",
            image=up_icon,
            compound="left",
            bg=t["surface_alt"],
            fg=t["text_bright"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=6,
            command=self.import_context_dialog,
            cursor="pointinghand"
        )
        btn_imp.pack(side="left", padx=(0, 6))

        trash_icon = get_tk_icon("trash", color="#FFFFFF", size=14)
        btn_reset = ActionButton(
            btn_r,
            text="  Reset Memory",
            image=trash_icon,
            compound="left",
            bg=t["danger_btn"],
            fg="#FFFFFF",
            hover_bg="#E11D48",
            active_bg="#BE123C",
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=6,
            command=self.flush_context_confirm,
            cursor="pointinghand"
        )
        btn_reset.pack(side="left")

        # Stats Card
        doc_count = self.get_document_count()
        stat_card = tk.Frame(container, bg=t["surface"], padx=16, pady=10, highlightbackground=t["border"], highlightthickness=1)
        stat_card.pack(fill="x", pady=(0, 10))

        tk.Label(stat_card, text=f"Active Profile: [{self.selected_profile.get()}]", font=("SF Pro Text", 11, "bold"), fg=t["primary"], bg=t["surface"]).pack(side="left")
        tk.Label(stat_card, text=f"  |  Total Indexed Context Fragments: {doc_count}", font=("SF Pro Text", 11), fg=t["muted"], bg=t["surface"]).pack(side="left", padx=10)

        # Scrollable Fragments Viewer
        scroll_frags = ScrollableFrame(container, bg=t["bg"])
        scroll_frags.pack(fill="both", expand=True)
        win = scroll_frags.scrollable_window

        chunks = self.get_profile_chunks()
        if not chunks:
            empty_card = tk.Frame(win, bg=t["surface"], padx=20, pady=36, highlightbackground=t["border"], highlightthickness=1)
            empty_card.pack(fill="x", pady=20)
            layers_big = get_tk_icon("layers", color=t["muted"], size=36)
            lbl_empty_layers = tk.Label(empty_card, image=layers_big, bg=t["surface"])
            lbl_empty_layers.image = layers_big
            lbl_empty_layers.pack()
            tk.Label(empty_card, text="No Memories Indexed for this Profile", font=("SF Pro Text", 13, "bold"), fg=t["text_bright"], bg=t["surface"]).pack(pady=(6, 2))
            tk.Label(
                empty_card,
                text="Click [Import Context] to parse your ChatGPT conversation export (JSON, ZIP, or Markdown).",
                font=("SF Pro Text", 10),
                fg=t["muted"],
                bg=t["surface"]
            ).pack()
        else:
            for idx, ch in enumerate(chunks):
                card = tk.Frame(win, bg=t["surface"], padx=12, pady=10, highlightbackground=t["border"], highlightthickness=1)
                card.pack(fill="x", pady=4)

                top_c = tk.Frame(card, bg=t["surface"])
                top_c.pack(fill="x", pady=(0, 4))
                tk.Label(top_c, text=f"Fragment #{idx+1}", font=("SF Pro Text", 8, "bold"), fg=t["primary"], bg=t["surface_alt"], padx=6, pady=1).pack(side="left")
                tk.Label(top_c, text=f"Profile: {self.selected_profile.get()}", font=("SF Pro Text", 8), fg=t["muted"], bg=t["surface"]).pack(side="left", padx=8)

                tk.Label(
                    card,
                    text=ch,
                    font=("SF Pro Text", 10),
                    fg=t["text"],
                    bg=t["surface"],
                    wraplength=940,
                    justify="left",
                    anchor="w"
                ).pack(fill="x")

    def on_profile_dropdown_changed(self, event=None):
        """Loads selected profile collection and refreshes profile view."""
        self.load_profile_collection(self.selected_profile.get())
        self.lbl_active_prof_pill.config(text=f"Profile: {self.selected_profile.get()}")
        self.show_page("profiles")

    # --- PAGE 3: REVIEW PACKETS INSPECTOR ---
    def build_packets_page(self):
        """Renders saved review packet file list and full markdown inspection reader."""
        t = self.theme
        container = tk.Frame(self.content_area, bg=t["bg"])
        container.pack(fill="both", expand=True)

        # Split: Left list (~320px) | Right Inspector
        left_list_frame = tk.Frame(container, bg=t["surface"], width=320, padx=14, pady=14, highlightbackground=t["border"], highlightthickness=1)
        left_list_frame.pack(side="left", fill="y", padx=(0, 10))
        left_list_frame.pack_propagate(False)

        top_l = tk.Frame(left_list_frame, bg=t["surface"])
        top_l.pack(fill="x", pady=(0, 10))

        tk.Label(top_l, text="Saved Review Packets", font=("SF Pro Text", 12, "bold"), fg=t["text_bright"], bg=t["surface"]).pack(side="left")
        ActionButton(
            top_l,
            text="  Refresh",
            image=get_tk_icon("refresh", color=t["text"], size=14),
            compound="left",
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 9, "bold"),
            padx=10,
            pady=4,
            command=self.refresh_review_packets_view,
            cursor="pointinghand"
        ).pack(side="right")

        self.packets_scroll = ScrollableFrame(left_list_frame, bg=t["surface"])
        self.packets_scroll.pack(fill="both", expand=True)
        self.packets_scroll.canvas.bind("<Button-1>", self._on_packet_list_click, add="+")

        # Right Inspector Frame
        self.right_packet_frame = tk.Frame(container, bg=t["surface"], padx=18, pady=16, highlightbackground=t["border"], highlightthickness=1)
        self.right_packet_frame.pack(side="right", fill="both", expand=True)

        self.render_packets_list()
        self.render_packet_inspector()

    def render_packets_list(self):
        """Lists exported markdown review packets with size and selection."""
        t = self.theme
        win = self.packets_scroll.scrollable_window
        for ch in win.winfo_children():
            ch.destroy()

        if not self.review_packets_list:
            tk.Label(win, text="No review packets found in\n~/.context_assistant/review_packets", font=("SF Pro Text", 10), fg=t["muted"], bg=t["surface"], justify="center").pack(pady=30)
            return

        for p_name in self.review_packets_list:
            is_sel = (p_name == self.selected_packet_name)
            p_path = os.path.join(REVIEWS_DIR, p_name)
            size_kb = os.path.getsize(p_path) / 1024 if os.path.exists(p_path) else 0

            card = tk.Frame(win, bg=t["surface_alt"] if is_sel else t["surface"], padx=10, pady=8, highlightbackground=t["primary"] if is_sel else t["border"], highlightthickness=1)
            card.pack(fill="x", pady=3)

            lbl_title = tk.Label(card, text=p_name, font=("SF Pro Text", 9, "bold" if is_sel else "normal"), fg=t["primary"] if is_sel else t["text"], bg=card.cget("bg"), anchor="w", cursor="pointinghand")
            lbl_title.pack(fill="x")

            lbl_sz = tk.Label(card, text=f"{size_kb:.1f} KB", font=("SF Pro Text", 8), fg=t["muted"], bg=card.cget("bg"), anchor="w", cursor="pointinghand")
            lbl_sz.pack(fill="x")

            # The title and size sit on top of the card, so the click has to
            # be bound on them as well as the frame.
            card._click_command = lambda name=p_name: self._open_review_packet(name)
            card.configure(cursor="pointinghand")
            for widget in (card, lbl_title, lbl_sz):
                widget.bind("<Button-1>", lambda _e, name=p_name: self._open_review_packet(name))
                self.packets_scroll.bind_mousewheel(widget)

    def _on_packet_list_click(self, event):
        """Select a packet when the canvas receives the click instead of the row."""
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        control = _control_from_widget(widget)
        command = getattr(control, "_click_command", None) if control is not None else None
        if command is not None:
            command()
            return "break"

    def _open_review_packet(self, name: str):
        """Open a packet after the click, so rebuilding the list does not eat the event."""
        now = time.monotonic()
        if getattr(self, "_packet_click_name", None) == name and now - getattr(self, "_packet_click_at", 0.0) < 0.3:
            return "break"
        self._packet_click_name = name
        self._packet_click_at = now
        self.root.after(0, lambda n=name: self.select_review_packet(n))
        return "break"

    def select_review_packet(self, name: str):
        """Sets active packet and displays file contents."""
        self.selected_packet_name = name
        self.render_packets_list()
        self.render_packet_inspector()

    def render_packet_inspector(self):
        """Displays full markdown text and copy sync prompt action."""
        t = self.theme
        for ch in self.right_packet_frame.winfo_children():
            ch.destroy()

        if not self.selected_packet_name:
            tk.Label(self.right_packet_frame, text="Select a review packet from the list on the left to inspect its contents.", font=("SF Pro Text", 11), fg=t["muted"], bg=t["surface"]).pack(pady=40)
            return

        p_path = os.path.join(REVIEWS_DIR, self.selected_packet_name)
        if not os.path.exists(p_path):
            tk.Label(self.right_packet_frame, text="Selected packet file does not exist.", font=("SF Pro Text", 11), fg=t["danger"], bg=t["surface"]).pack(pady=40)
            return

        try:
            with open(p_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"Could not read packet file: {e}"

        hdr = tk.Frame(self.right_packet_frame, bg=t["surface"])
        hdr.pack(fill="x", pady=(0, 10))

        tk.Label(hdr, text=self.selected_packet_name, font=("SF Pro Text", 12, "bold"), fg=t["text_bright"], bg=t["surface"]).pack(side="left")

        copy_icon = get_tk_icon("copy", color="#0F172A", size=14)
        btn_sync = ActionButton(
            hdr,
            text="  Copy Sync Prompt",
            image=copy_icon,
            compound="left",
            bg=t["primary"],
            fg="#0F172A" if self.theme_name == "midnight" else "#FFFFFF",
            hover_bg=t["primary_hover"],
            active_bg=t["primary_light"],
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=6,
            command=lambda: self.copy_sync_prompt(content),
            cursor="pointinghand"
        )
        btn_sync.pack(side="right", padx=(6, 0))

        btn_export = ActionButton(
            hdr,
            text="  Export",
            image=get_tk_icon("file_text", color=t["text"], size=14),
            compound="left",
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=6,
            command=lambda: self.open_export_dialog("packet"),
            cursor="pointinghand"
        )
        btn_export.pack(side="right", padx=(6, 0))

        trash_icon = get_tk_icon("trash", color="#FFFFFF", size=14)
        btn_del = ActionButton(
            hdr,
            text="  Delete",
            image=trash_icon,
            compound="left",
            bg=t["danger_btn"],
            fg="#FFFFFF",
            hover_bg="#E11D48",
            active_bg="#BE123C",
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=6,
            command=lambda: self.delete_packet(self.selected_packet_name),
            cursor="pointinghand"
        )
        btn_del.pack(side="right")

        txt_frame = tk.Frame(self.right_packet_frame, bg=t["surface_alt"], bd=1, relief="solid")
        txt_frame.pack(fill="both", expand=True)

        text_widget = tk.Text(
            txt_frame,
            bg=t["surface_alt"],
            fg=t["text"],
            insertbackground=t["text"],
            wrap="word",
            font=("SF Mono", 10),
            padx=12,
            pady=12,
            bd=0
        )
        text_widget.insert("1.0", content)
        text_widget.config(state="disabled")

        scrollbar = ttk.Scrollbar(txt_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def copy_sync_prompt(self, packet_content: str):
        """Extracts UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT block and copies to clipboard."""
        match = re.search(r"```text\n(UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT[\s\S]*?)```", packet_content)
        prompt_text = match.group(1).strip() if match else packet_content.strip()

        try:
            subprocess.run(["pbcopy"], input=prompt_text.encode("utf-8"), check=True)
            messagebox.showinfo("Copied", "ChatGPT memory sync prompt copied to clipboard!\nPaste it directly into your ChatGPT thread.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy prompt: {e}")

    def delete_packet(self, name: str):
        """Deletes review packet after confirmation."""
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {name}?")
        if confirm:
            p_path = os.path.join(REVIEWS_DIR, name)
            if os.path.exists(p_path):
                os.remove(p_path)
            self.refresh_review_packets()
            self.selected_packet_name = None
            self.render_packets_list()
            self.render_packet_inspector()

    def refresh_review_packets_view(self):
        """Scans disk for packets and redraws view."""
        self.refresh_review_packets()
        self.render_packets_list()

    # --- PAGE 4: SETTINGS & MODELS CONSOLE (SCROLLABLE) ---
    def build_settings_page(self):
        """Renders comprehensive settings: dynamic model discovery, language, STT, prompt tuning, and restore defaults."""
        t = self.theme
        scroll_settings = ScrollableFrame(self.content_area, bg=t["bg"])
        scroll_settings.pack(fill="both", expand=True)
        win = scroll_settings.scrollable_window

        # Section 1: LLM Inference Engine & Remote Model Discovery
        card_llm = tk.Frame(win, bg=t["surface"], padx=18, pady=16, highlightbackground=t["border"], highlightthickness=1)
        card_llm.pack(fill="x", pady=(0, 14))

        tk.Label(card_llm, text="1. LLM Inference Engine & Dynamic Model Discovery", font=("SF Pro Text", 13, "bold"), fg=t["text_bright"], bg=t["surface"]).pack(anchor="w", pady=(0, 10))

        # Provider Selector
        row_prov = tk.Frame(card_llm, bg=t["surface"])
        row_prov.pack(fill="x", pady=4)
        tk.Label(row_prov, text="LLM Provider:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")
        combo_prov = SelectField(row_prov, textvariable=self.llm_provider, values=["ollama", "openai", "custom"], state="readonly", width=22, bg=t["input_bg"], fg=t["text"], border=t["border"], accent=t["primary"], muted=t["muted"])
        combo_prov.pack(side="left")
        combo_prov.bind("<<ComboboxSelected>>", lambda e: self.on_provider_changed())

        # Ollama URL
        self.row_ollama = tk.Frame(card_llm, bg=t["surface"])
        self.row_ollama.pack(fill="x", pady=4)
        tk.Label(self.row_ollama, text="Ollama Server URL:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")
        entry_ollama = tk.Entry(self.row_ollama, textvariable=self.ollama_url_input, bg=t["input_bg"], fg=t["text"], insertbackground=t["text"], width=35)
        entry_ollama.pack(side="left")

        # OpenAI API Key
        self.row_openai = tk.Frame(card_llm, bg=t["surface"])
        self.row_openai.pack(fill="x", pady=4)
        tk.Label(self.row_openai, text="OpenAI API Key:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")
        self.entry_openai = tk.Entry(self.row_openai, textvariable=self.openai_key_input, show="*", bg=t["input_bg"], fg=t["text"], insertbackground=t["text"], width=35)
        self.entry_openai.pack(side="left", padx=(0, 8))

        btn_toggle_key = ActionButton(
            self.row_openai,
            text="  Show",
            image=get_tk_icon("eye", color=t["text"], size=14),
            compound="left",
            bg=t["surface_alt"],
            fg=t["text"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 9, "bold"),
            padx=10,
            pady=4,
            command=self.toggle_openai_key_mask,
            cursor="pointinghand"
        )
        btn_toggle_key.pack(side="left")

        # Custom Endpoint URL & Key
        self.row_custom = tk.Frame(card_llm, bg=t["surface"])
        self.row_custom.pack(fill="x", pady=4)
        tk.Label(self.row_custom, text="Custom Endpoint URL:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")
        entry_custom = tk.Entry(self.row_custom, textvariable=self.custom_url_input, bg=t["input_bg"], fg=t["text"], insertbackground=t["text"], width=35)
        entry_custom.pack(side="left")

        # Dynamic Model Selector & Fetch Button
        row_model = tk.Frame(card_llm, bg=t["surface"])
        row_model.pack(fill="x", pady=(10, 4))
        tk.Label(row_model, text="Active LLM Model:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")

        model_choices = self.discovered_models if self.discovered_models else [self.selected_model.get() or "llama3.1:8b"]
        self.combo_models = SelectField(row_model, textvariable=self.selected_model, values=model_choices, state="readonly" if self.discovered_models else "normal", width=26, bg=t["input_bg"], fg=t["text"], border=t["border"], accent=t["primary"], muted=t["muted"])
        self.combo_models.pack(side="left", padx=(0, 10))

        refresh_icon = get_tk_icon("refresh", color=t["text_bright"], size=14)
        btn_fetch = ActionButton(
            row_model,
            text="  Fetch Models",
            image=refresh_icon,
            compound="left",
            bg=t["surface_alt"],
            fg=t["text_bright"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 9, "bold"),
            padx=12,
            pady=4,
            command=self.fetch_available_models_ui,
            cursor="pointinghand"
        )
        btn_fetch.pack(side="left")

        self.lbl_models_status = tk.Label(card_llm, text=f"Active Model: {self.selected_model.get()}", font=("SF Pro Text", 9), fg=t["muted"], bg=t["surface"])
        self.lbl_models_status.pack(anchor="w", pady=(4, 0))

        # Section 2: Speech-to-Text (STT) Settings
        card_stt = tk.Frame(win, bg=t["surface"], padx=18, pady=16, highlightbackground=t["border"], highlightthickness=1)
        card_stt.pack(fill="x", pady=(0, 14))

        tk.Label(card_stt, text="2. Speech-to-Text & Multilingual Configuration", font=("SF Pro Text", 13, "bold"), fg=t["text_bright"], bg=t["surface"]).pack(anchor="w", pady=(0, 10))

        # Language Selector
        row_lang = tk.Frame(card_stt, bg=t["surface"])
        row_lang.pack(fill="x", pady=4)
        tk.Label(row_lang, text="Spoken Language:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")

        lang_tags = [code for code, _ in LANGUAGE_OPTIONS]
        self.combo_lang = SelectField(row_lang, textvariable=self.stt_language, values=lang_tags, state="readonly", width=22, bg=t["input_bg"], fg=t["text"], border=t["border"], accent=t["primary"], muted=t["muted"])
        self.combo_lang.pack(side="left", padx=(0, 10))
        self.combo_lang.bind("<<ComboboxSelected>>", self.on_language_dropdown_changed)

        # STT Whisper Model
        row_whisper = tk.Frame(card_stt, bg=t["surface"])
        row_whisper.pack(fill="x", pady=4)
        tk.Label(row_whisper, text="Whisper Model:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")

        is_en = (self.stt_language.get() == "en")
        current_model_choices = [tag for tag, _ in (ENGLISH_MODELS if is_en else MULTILINGUAL_MODELS)]
        self.combo_whisper = SelectField(row_whisper, textvariable=self.whisper_model_option, values=current_model_choices, state="readonly", width=22, bg=t["input_bg"], fg=t["text"], border=t["border"], accent=t["primary"], muted=t["muted"])
        self.combo_whisper.pack(side="left")
        self.combo_whisper.bind("<<ComboboxSelected>>", self.on_whisper_model_dropdown_changed)

        # Two capture sources: the local microphone and the other people on the call.
        row_mic = tk.Frame(card_stt, bg=t["surface"])
        row_mic.pack(fill="x", pady=4)
        tk.Label(row_mic, text="Your microphone:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")
        mic_labels = dict(self.mic_devices)
        self.combo_mic = SelectField(
            row_mic,
            textvariable=self.selected_mic,
            values=[val for val, _ in self.mic_devices],
            labels=mic_labels,
            state="readonly",
            width=28,
            bg=t["input_bg"],
            fg=t["text"],
            border=t["border"],
            accent=t["primary"],
            muted=t["muted"],
        )
        self.combo_mic.pack(side="left", padx=(0, 10))
        self.combo_mic.bind("<<ComboboxSelected>>", lambda e: self.on_audio_device_changed())

        row_meet = tk.Frame(card_stt, bg=t["surface"])
        row_meet.pack(fill="x", pady=4)
        tk.Label(row_meet, text="Meeting audio:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"], width=18, anchor="w").pack(side="left")
        meeting_labels = dict(self.meeting_devices)
        self.combo_meeting = SelectField(
            row_meet,
            textvariable=self.selected_meeting,
            values=[val for val, _ in self.meeting_devices],
            labels=meeting_labels,
            state="readonly",
            width=28,
            bg=t["input_bg"],
            fg=t["text"],
            border=t["border"],
            accent=t["primary"],
            muted=t["muted"],
        )
        self.combo_meeting.pack(side="left", padx=(0, 10))
        self.combo_meeting.bind("<<ComboboxSelected>>", lambda e: self.on_audio_device_changed())

        # Section 3: Prompt & Generation Tuning Parameters
        card_prompt = tk.Frame(win, bg=t["surface"], padx=18, pady=16, highlightbackground=t["border"], highlightthickness=1)
        card_prompt.pack(fill="x", pady=(0, 14))

        tk.Label(card_prompt, text="3. System Prompt Template & Generation Parameters", font=("SF Pro Text", 13, "bold"), fg=t["text_bright"], bg=t["surface"]).pack(anchor="w", pady=(0, 6))
        tk.Label(card_prompt, text="Use {context} as placeholder where verified ChromaDB work history chunks are injected.", font=("SF Pro Text", 9), fg=t["muted"], bg=t["surface"]).pack(anchor="w", pady=(0, 8))

        self.txt_prompt = tk.Text(card_prompt, bg=t["input_bg"], fg=t["text"], insertbackground=t["text"], height=7, font=("SF Mono", 10), wrap="word", bd=1, relief="solid")
        self.txt_prompt.pack(fill="x", pady=(0, 10))
        self.txt_prompt.insert("1.0", self.system_prompt_text)

        # Tuning Sliders / Entry Row
        row_tuning = tk.Frame(card_prompt, bg=t["surface"])
        row_tuning.pack(fill="x", pady=4)

        tk.Label(row_tuning, text="Temperature:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"]).pack(side="left", padx=(0, 4))
        tk.Entry(row_tuning, textvariable=self.temperature, width=6, bg=t["input_bg"], fg=t["text"], insertbackground=t["text"]).pack(side="left", padx=(0, 16))

        tk.Label(row_tuning, text="Max Tokens:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"]).pack(side="left", padx=(0, 4))
        tk.Entry(row_tuning, textvariable=self.max_tokens, width=6, bg=t["input_bg"], fg=t["text"], insertbackground=t["text"]).pack(side="left", padx=(0, 16))

        tk.Label(row_tuning, text="Silence Threshold:", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"]).pack(side="left", padx=(0, 4))
        tk.Entry(row_tuning, textvariable=self.silence_threshold, width=6, bg=t["input_bg"], fg=t["text"], insertbackground=t["text"]).pack(side="left")

        # Action Buttons Row
        actions_bar = tk.Frame(win, bg=t["bg"])
        actions_bar.pack(fill="x", pady=(6, 20))

        save_icon = get_tk_icon("save", color="#0F172A", size=14)
        btn_save = ActionButton(
            actions_bar,
            text="  Save Settings",
            image=save_icon,
            compound="left",
            bg=t["primary"],
            fg="#0F172A" if self.theme_name == "midnight" else "#FFFFFF",
            hover_bg=t["primary_hover"],
            active_bg=t["primary_light"],
            font=("SF Pro Text", 10, "bold"),
            padx=14,
            pady=8,
            command=self.save_settings_from_ui,
            cursor="pointinghand"
        )
        btn_save.pack(side="left", padx=(0, 8))

        reset_icon = get_tk_icon("rotate_ccw", color=t["text_bright"], size=14)
        btn_reset_p = ActionButton(
            actions_bar,
            text="  Reset System Prompt",
            image=reset_icon,
            compound="left",
            bg=t["surface_alt"],
            fg=t["text_bright"],
            hover_bg=t["border"],
            active_bg=t["surface"],
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=8,
            command=self.reset_system_prompt_to_default,
            cursor="pointinghand"
        )
        btn_reset_p.pack(side="left", padx=(0, 8))

        warn_icon = get_tk_icon("alert_triangle", color="#FFFFFF", size=14)
        btn_reset_all = ActionButton(
            actions_bar,
            text="  Restore All Defaults",
            image=warn_icon,
            compound="left",
            bg=t["danger_btn"],
            fg="#FFFFFF",
            hover_bg="#E11D48",
            active_bg="#BE123C",
            font=("SF Pro Text", 10, "bold"),
            padx=12,
            pady=8,
            command=self.reset_all_to_defaults,
            cursor="pointinghand"
        )
        btn_reset_all.pack(side="right")

    def toggle_openai_key_mask(self):
        """Toggles password masking on OpenAI API key entry."""
        if self.entry_openai.cget("show") == "":
            self.entry_openai.config(show="*")
        else:
            self.entry_openai.config(show="")

    def on_provider_changed(self):
        """Updates model dropdown and provider client when provider changes."""
        self.init_llm_client()
        self.fetch_available_models_ui()

    def on_language_dropdown_changed(self, event=None):
        """Switches available models between English and Multilingual."""
        is_en = (self.stt_language.get() == "en")
        new_choices = [tag for tag, _ in (ENGLISH_MODELS if is_en else MULTILINGUAL_MODELS)]
        self.combo_whisper["values"] = new_choices

        cur = self.whisper_model_option.get()
        if cur not in new_choices:
            new_model = "small.en" if is_en else "small"
            self.whisper_model_option.set(new_model)
        self.init_whisper_engine()
        self.save_config()

    def on_whisper_model_dropdown_changed(self, event=None):
        """Re-initializes whisper model on user selection."""
        self.save_config()
        self.init_whisper_engine()

    def on_audio_device_changed(self):
        """Remembers the two audio sources and refreshes the header."""
        self._apply_audio_selection_label()
        self.save_config()
        self.update_header()

    def fetch_available_models_ui(self):
        """Triggers dynamic model discovery in background and updates combobox in Settings."""
        if hasattr(self, "lbl_models_status") and self.lbl_models_status.winfo_exists():
            self.lbl_models_status.config(text="Fetching remote models...")

        def worker():
            models = self.fetch_available_models()
            def update_ui():
                if hasattr(self, "combo_models") and self.combo_models.winfo_exists():
                    if models:
                        self.combo_models["values"] = models
                        if self.selected_model.get() not in models:
                            self.selected_model.set(models[0])
                        if hasattr(self, "lbl_models_status") and self.lbl_models_status.winfo_exists():
                            self.lbl_models_status.config(text=f"Discovered {len(models)} models. Active: {self.selected_model.get()}")
                    else:
                        if hasattr(self, "lbl_models_status") and self.lbl_models_status.winfo_exists():
                            self.lbl_models_status.config(text="No models returned from provider server.")
            self.safe_after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    def save_settings_from_ui(self):
        """Captures prompt text and saves config to ~/.context_assistant/config.json."""
        self.system_prompt_text = self.txt_prompt.get("1.0", "end-1c").strip()
        self.save_config()
        messagebox.showinfo("Saved", "Settings saved successfully to config.json!")

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
        self.llm_model_name = self.selected_model.get()

        loaded_prompt = cfg.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
        if not loaded_prompt or ("CRITICAL OPERATIONAL RULES:" in loaded_prompt and "DIRECT MEMORY ANCHORING" not in loaded_prompt):
            loaded_prompt = DEFAULT_SYSTEM_PROMPT
        self.system_prompt_text = loaded_prompt
        loaded_temp = cfg.get("temperature", DEFAULT_CONFIG["temperature"])
        if loaded_temp == 0.7:
            loaded_temp = 0.0
        self.temperature.set(str(loaded_temp))
        self.max_tokens.set(str(cfg.get("max_tokens", DEFAULT_CONFIG["max_tokens"])))
        self.silence_threshold.set(str(cfg.get("silence_threshold", DEFAULT_CONFIG["silence_threshold"])))
        self.stt_language.set(cfg.get("stt_language", DEFAULT_CONFIG["stt_language"]))
        self.whisper_model_option.set(cfg.get("whisper_model", DEFAULT_CONFIG["whisper_model"]))
        mic_choice, meeting_choice = self._audio_choices_from_config(cfg)
        self.selected_mic.set(mic_choice)
        self.selected_meeting.set(meeting_choice)

    def save_config(self):
        """Saves current provider, URLs, credentials, model, system prompt, and parameters to config.json."""
        try:
            temp_val = float(str(self.temperature.get()).strip())
        except Exception:
            temp_val = 0.0

        try:
            tokens_val = int(str(self.max_tokens.get()).strip())
        except Exception:
            tokens_val = 150

        try:
            vad_val = float(str(self.silence_threshold.get()).strip())
        except Exception:
            vad_val = 0.015

        data = {
            "provider": self.llm_provider.get(),
            "ollama_url": self.ollama_url_input.get().strip(),
            "openai_api_key": self.openai_key_input.get().strip(),
            "custom_url": self.custom_url_input.get().strip(),
            "custom_api_key": self.custom_key_input.get().strip(),
            "selected_model": self.selected_model.get().strip(),
            "stt_language": self.stt_language.get(),
            "whisper_model": self.whisper_model_option.get().strip(),
            "system_prompt": self.system_prompt_text,
            "temperature": temp_val,
            "max_tokens": tokens_val,
            "silence_threshold": vad_val,
            "mic_device": self.selected_mic.get() or "auto",
            "meeting_device": self.selected_meeting.get() or "auto",
        }

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("[Config Engine] Persisted settings to ~/.context_assistant/config.json")
        except Exception as e:
            print(f"[Config Save Error]: {e}", file=sys.stderr)

    def reset_system_prompt_to_default(self, silent: bool = False):
        """Restores default system prompt with zero-hallucination constraint."""
        self.system_prompt_text = DEFAULT_CONFIG["system_prompt"]
        if hasattr(self, "txt_prompt"):
            self.txt_prompt.delete("1.0", "end")
            self.txt_prompt.insert("1.0", self.system_prompt_text)
        self.save_config()
        if not silent:
            messagebox.showinfo("Reset", "System prompt restored to factory default.")

    def reset_all_to_defaults(self, silent: bool = False):
        """Restores complete application defaults."""
        if not silent:
            confirm = messagebox.askyesno("Restore Defaults", "Are you sure you want to restore all application settings to defaults?")
            if not confirm:
                return

        for k, v in DEFAULT_CONFIG.items():
            if k == "provider":
                self.llm_provider.set(v)
            elif k == "ollama_url":
                self.ollama_url_input.set(v)
            elif k == "openai_api_key":
                self.openai_key_input.set(v)
            elif k == "custom_url":
                self.custom_url_input.set(v)
            elif k == "custom_api_key":
                self.custom_key_input.set(v)
            elif k == "selected_model":
                self.selected_model.set(v)
            elif k == "stt_language":
                self.stt_language.set(v)
            elif k == "whisper_model":
                self.whisper_model_option.set(v)
            elif k == "system_prompt":
                self.system_prompt_text = v
            elif k == "temperature":
                self.temperature.set(str(v))
            elif k == "max_tokens":
                self.max_tokens.set(str(v))
            elif k == "silence_threshold":
                self.silence_threshold.set(str(v))
            elif k == "mic_device":
                self.selected_mic.set(str(v))
            elif k == "meeting_device":
                self.selected_meeting.set(str(v))

        self.save_config()
        self.init_llm_client()
        self.init_whisper_engine()
        self.show_page("settings")
        if not silent:
            messagebox.showinfo("Defaults Restored", "All application settings restored to factory defaults.")

    # --- ENGINES INITIALIZATION ---
    def init_chromadb(self):
        """Initializes ChromaDB vector database."""
        try:
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            self.load_profile_collection(self.selected_profile.get())
        except Exception as e:
            print(f"[ChromaDB Init Error]: {e}", file=sys.stderr)
            self.collection = None

    def load_profile_collection(self, profile_name: str):
        """Loads collection for profile."""
        if not self.chroma_client:
            return
        safe_name = "prof_" + re.sub(r'[^a-zA-Z0-9_-]', '_', profile_name).lower()[:50]
        try:
            self.collection = self.chroma_client.get_or_create_collection(name=safe_name)
            cnt = self.collection.count()
            self.suggestion_text = (
                f"Active Profile: [{profile_name}]\n"
                f"Knowledge Memory: {cnt} indexed context chunks.\n\n"
                f"Start call session to view real-time grounded suggestions."
            )
        except Exception as e:
            print(f"[ChromaDB Collection Error]: {e}", file=sys.stderr)

    def init_llm_client(self):
        """Initializes OpenAI client targeting Ollama, OpenAI, or Custom endpoint."""
        provider = self.llm_provider.get()
        selected_model = self.selected_model.get().strip()

        if provider == "openai":
            key = self.openai_key_input.get().strip() or os.getenv("OPENAI_API_KEY", "")
            self.llm_client = OpenAI(api_key=key or "none")
            self.llm_model_name = selected_model or "gpt-4o-mini"
            self.engine_model_label = f"OpenAI ({self.llm_model_name})"
        elif provider == "custom":
            raw_url = self.custom_url_input.get().strip().rstrip("/")
            key = self.custom_key_input.get().strip() or "none"
            self.llm_client = OpenAI(base_url=raw_url, api_key=key)
            self.llm_model_name = selected_model or "default"
            self.engine_model_label = f"Custom ({self.llm_model_name})"
        else:
            # Ollama
            ollama_url = self.ollama_url_input.get().strip().rstrip("/")
            host = ollama_url[:-3] if ollama_url.endswith("/v1") else ollama_url
            sdk_url = f"{host}/v1"
            self.llm_client = OpenAI(base_url=sdk_url, api_key="ollama")
            self.llm_model_name = selected_model or "llama3.1:8b"
            self.engine_model_label = f"Ollama ({self.llm_model_name})"

    def init_whisper_engine(self):
        """Initializes Faster-Whisper fast & accurate model in background."""
        model_tag = self.whisper_model_option.get() or "small.en"
        stt_lang = self.stt_language.get() or "en"
        self._active_model_tag = model_tag
        self._active_stt_lang = stt_lang

        def load():
            with self._whisper_lock:
                if self.whisper_model is not None and self._loaded_whisper_tag == model_tag:
                    return

                try:
                    self.status_text = f"LOADING STT ({model_tag})"
                    self.status_color = "#7DD3FC"
                    self.safe_after(0, self.update_header)

                    print(f"[STT Engine] Loading Faster-Whisper {model_tag}...")
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
                    self.status_text = "SYSTEM READY"
                    self.status_color = "#34D399"
                    self.safe_after(0, self.update_header)
                except Exception as e:
                    print(f"[STT Error] Whisper {model_tag} loading failed: {e}", file=sys.stderr)
                    try:
                        fallback_tag = "base.en" if stt_lang == "en" else "base"
                        self.whisper_model = WhisperModel(fallback_tag, device="cpu", compute_type="int8")
                        self._loaded_whisper_tag = fallback_tag
                        self.status_text = f"SYSTEM READY ({fallback_tag.upper()})"
                        self.status_color = "#34D399"
                        self.safe_after(0, self.update_header)
                    except Exception as ex2:
                        self.status_text = "STT FAILED"
                        self.status_color = "#FDA4AF"
                        self.safe_after(0, self.update_header)

        threading.Thread(target=load, daemon=True).start()

    def fetch_available_models(self) -> List[str]:
        """Queries remote Ollama tags or OpenAI models."""
        provider = self.llm_provider.get()
        models = []

        if provider == "ollama":
            url = self.ollama_url_input.get().strip().rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            try:
                import urllib.request
                req = urllib.request.Request(f"{url}/api/tags", headers={"User-Agent": "AI-Assistant/2.0"})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    data = json.loads(resp.read().decode())
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            except Exception as e:
                print(f"[Ollama Tags Error]: {e}", file=sys.stderr)

        elif provider in ("openai", "custom"):
            try:
                if self.llm_client:
                    resp = self.llm_client.models.list()
                    models = [m.id for m in resp.data]
            except Exception as e:
                print(f"[OpenAI Models List Error]: {e}", file=sys.stderr)

        self.discovered_models = models
        return models

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
        self.mic_devices = [("auto", "System default microphone")] + physical + loopbacks
        self.meeting_devices = [("auto", "BlackHole if available")] + multi_outputs + loopbacks + physical
        mic_ids = {val for val, _ in self.mic_devices}
        meeting_ids = {val for val, _ in self.meeting_devices}
        if self.selected_mic.get() not in mic_ids:
            self.selected_mic.set("auto")
        if self.selected_meeting.get() not in meeting_ids:
            self.selected_meeting.set("auto")
        self._apply_audio_selection_label()
        self.update_header()

    def _audio_choices_from_config(self, cfg: Dict[str, Any]) -> Tuple[str, str]:
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
            self.active_device_label = "No input devices"
            return
        self.active_device_label = " + ".join(
            "Mic" if s["speaker"] == "You" else "Meeting" for s in sources
        )

    @staticmethod
    def _is_loopback_device(name: str) -> bool:
        return "blackhole" in (name or "").lower()

    def resolve_capture_sources(self) -> List[Dict[str, Any]]:
        """
        Opens the microphone chosen in Settings and the meeting source chosen
        beside it. Auto uses the system microphone and BlackHole when it exists.
        """
        devices = list(sd.query_devices())
        inputs = [(idx, dev) for idx, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        if not inputs:
            raise RuntimeError("No input audio devices found.")

        microphones = [(idx, dev) for idx, dev in inputs if not self._is_loopback_device(dev['name'])]

        mic = self._lookup_input_device(self.selected_mic.get())
        loop = resolve_meeting_device(self.selected_meeting.get())

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
    def toggle_session(self):
        """Starts or stops live call monitoring."""
        if self.is_listening:
            self.stop_session()
        else:
            self.start_session()

    def audio_callback(self, indata, frames, time_info, status):
        """Single-stream fallback used by diagnostics. Live sessions use per-source callbacks."""
        if status:
            print(f"[Audio Status] {status}", file=sys.stderr)
        self.audio_queue.put(("Participant", self.current_samplerate or 16000, to_mono(indata)))

    def start_session(self):
        """Opens the microphone and BlackHole together, then starts the STT loop."""
        if self.is_listening:
            return

        if self.whisper_model is None:
            messagebox.showinfo("Starting", "Whisper STT is still loading. Please wait a moment...")
            return

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self._active_silence_threshold = float(str(self.silence_threshold.get()).strip())
        except Exception:
            self._active_silence_threshold = 0.015

        try:
            self._active_temperature = float(str(self.temperature.get()).strip())
        except Exception:
            self._active_temperature = 0.0

        try:
            self._active_max_tokens = int(str(self.max_tokens.get()).strip())
        except Exception:
            self._active_max_tokens = 150

        self._active_stt_lang = self.stt_language.get() or "en"

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
                detail = "\n".join(errors) or "No input stream could be started."
                raise RuntimeError(detail)

            self.streams = opened
            self.stream = opened[0]
            live_sources = [s for s in sources if "samplerate" in s]
            self._has_meeting_stream = any(s["speaker"] == "Participant" for s in live_sources)
            self.current_samplerate = live_sources[0]["samplerate"]
            names = [f"{s['speaker']}: {s['info']['name']}" for s in live_sources]
            rates = [f"{s['speaker']} {s['samplerate']}Hz" for s in live_sources]
            self.active_device_label = " + ".join(
                "BlackHole" if self._is_loopback_device(s["info"]["name"]) else "Mic" for s in live_sources
            )
            self.hardware_sr_label = " · ".join(rates) + " (-> 16kHz)"
            self._capture_summary = " | ".join(names)

            self.is_listening = True
            self.status_text = "LIVE CALL ACTIVE"
            self.status_color = "#FDA4AF"
            with self._history_lock:
                self._stt_generation += 1
                self.transcript_history.clear()
                self.suggestion_history.clear()

            self._start_stt_worker()
            self.processing_thread = threading.Thread(target=self.process_audio_pipeline, daemon=True)
            self.processing_thread.start()
            self.update_header()
            if errors:
                messagebox.showwarning("Audio", "One source could not be opened:\n" + "\n".join(errors))

        except Exception as e:
            self._close_audio_streams()
            print(f"[Audio Stream Start Error]: {e}", file=sys.stderr)
            messagebox.showerror("Audio Error", f"Could not open audio devices:\n{e}")
            if self.is_listening:
                self.stop_session()

    def stop_session(self):
        """Stops both audio streams, exports the review packet, and returns to idle."""
        if not self.is_listening:
            return

        self.is_listening = False
        self._close_audio_streams()
        vad_thread = getattr(self, "processing_thread", None)
        if vad_thread is not None and vad_thread.is_alive():
            vad_thread.join(timeout=1.0)
        self._stop_stt_worker()

        self.status_text = "SESSION STOPPED"
        self.status_color = "#94A3B8"
        self.update_header()

    def _vad_take_phrase(self, state: Dict[str, Any], chunk: np.ndarray):
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
                self.process_phrase(phrase, sr, speaker, generation)
            except Exception as exc:
                print(f"[Transcription Error]: {exc}", file=sys.stderr)

    def _submit_phrase(self, phrase: np.ndarray, sr: int, speaker: str):
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
        """Ask the worker to finish queued phrases, then save, without freezing the window."""
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
            self.safe_after(0, self._publish_stopped_session)

        threading.Thread(target=finish, daemon=True).start()

    def _publish_stopped_session(self):
        self.export_post_call_packet()
        self.refresh_review_packets()
        if self.active_page == "packets":
            self.render_packets_list()
            self.render_packet_inspector()

    def _is_echo(self, speaker: str, text: str) -> bool:
        previous = self.transcript_history[-1] if self.transcript_history else None
        if previous is None:
            return False
        _stamp, previous_speaker, previous_text = previous
        return is_cross_talk_echo(previous_speaker, previous_text, speaker, text)

    def process_audio_pipeline(self):
        """Buffers each source separately so your voice and theirs are transcribed apart."""
        cfg_thresh = getattr(self, "_active_silence_threshold", 0.015)
        states: Dict[str, Dict[str, Any]] = {}
        print(f"[Audio Pipeline] Dual-source VAD (baseline={cfg_thresh})")

        while self.is_listening:
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

    def process_phrase(self, audio_data: np.ndarray, samplerate: Optional[int] = None, speaker: str = "Participant", generation: Optional[int] = None):
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
            lang = getattr(self, "_active_stt_lang", "en")
            phrase_text = transcribe_speech(model, audio_16k, lang, getattr(self, "_loaded_whisper_tag", ""))
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
                print(f"[Transcription] [{timestamp}] {speaker}: {phrase_text}")
                self.safe_after(0, self._show_transcript, timestamp, speaker, phrase_text)

        except Exception as e:
            print(f"[Transcription Error]: {e}", file=sys.stderr)

    def _show_transcript(self, timestamp: str, speaker: str, phrase_text: str):
        """Apply a finished transcript on the Tk thread."""
        self.transcripts.insert(0, {"time": timestamp, "speaker": speaker, "text": phrase_text})
        if self.active_page == "copilot":
            self.render_transcripts_view()
        if speaker != "You" or not self._has_meeting_stream:
            print(f"[Suggestion Engine] Querying context for: '{phrase_text}'")
            threading.Thread(
                target=self.generate_contextual_suggestion, args=(phrase_text, speaker), daemon=True
            ).start()

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
                    if hasattr(self, "transcripts") and self.transcripts:
                        prev_text = self.transcripts[0].get("text", "").strip()
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

        self.retrieved_sources = sources

        raw_prompt = self.system_prompt_text.strip() or DEFAULT_SYSTEM_PROMPT
        if "{context}" in raw_prompt:
            final_system_prompt = raw_prompt.replace("{context}", context_str)
        else:
            final_system_prompt = f"{raw_prompt}\n\n=== VERIFIED CLIENT / WORK HISTORY ===\n{context_str}"

        temp_val = getattr(self, "_active_temperature", 0.0)
        tokens_val = getattr(self, "_active_max_tokens", 160)

        # Construct user prompt with recent live call context if available
        prior_turns = []
        try:
            if hasattr(self, "transcripts") and self.transcripts:
                recent_items = self.transcripts[:2]
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
            self.suggestion_text = suggestion
            self.suggestion_history.append((time.strftime("%H:%M:%S"), speaker, suggestion))
            print(f"[Suggestion Generated]: {suggestion}")

            if self.active_page == "copilot":
                self.safe_after(0, self.render_suggestions_view)

        except Exception as e:
            err_msg = (
                f"LLM Connection Error:\n"
                f"Ensure your provider server ({self.llm_provider.get()}) is online.\n\nDetails: {e}"
            )
            self.suggestion_text = err_msg
            print(f"[Suggestion Error]: {e}", file=sys.stderr)
            if self.active_page == "copilot":
                self.safe_after(0, self.render_suggestions_view)

    # --- PROFILES & INGESTION ENGINE ---
    def refresh_profiles(self):
        """Scans profiles directory."""
        subdirs = self.get_profile_names()
        cur = self.selected_profile.get()
        if not cur or cur not in subdirs:
            self.selected_profile.set(subdirs[0])
            self.load_profile_collection(subdirs[0])

    def get_profile_names(self) -> List[str]:
        """Lists profile directory subfolders."""
        subdirs = [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d)) and not d.startswith('.')]
        if not subdirs:
            default_path = os.path.join(PROFILES_DIR, "Default_Client")
            os.makedirs(default_path, exist_ok=True)
            subdirs = ["Default_Client"]
        subdirs.sort()
        return subdirs

    def get_document_count(self) -> int:
        if self.collection:
            try:
                return self.collection.count()
            except Exception:
                return 0
        return 0

    def get_profile_chunks(self) -> List[str]:
        if not self.collection:
            return []
        try:
            results = self.collection.get()
            return results.get("documents", []) or []
        except Exception:
            return []

    def create_new_profile_dialog(self):
        """Modal dialog to create and switch to a new client profile."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New Profile")
        dialog.geometry("400x170")
        dialog.configure(bg=self.theme["surface"])
        dialog.transient(self.root)
        def on_close():
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)
        dialog.grab_set()

        lbl = tk.Label(dialog, text="Enter Profile Name (e.g. Acme_Corp):", font=("SF Pro Text", 11, "bold"), fg=self.theme["text_bright"], bg=self.theme["surface"])
        lbl.pack(pady=(20, 8))

        entry = tk.Entry(dialog, width=30, font=("SF Pro Text", 11), bg=self.theme["input_bg"], fg=self.theme["text"], insertbackground=self.theme["text"])
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
            self.selected_profile.set(name_clean)
            self.load_profile_collection(name_clean)
            self.lbl_active_prof_pill.config(text=f"Profile: {name_clean}")
            on_close()
            self.show_page("profiles")

        btn = ActionButton(
            dialog,
            text="  Create Profile",
            image=get_tk_icon("plus", color="#0F172A" if self.theme_name == "midnight" else "#FFFFFF", size=14),
            compound="left",
            bg=self.theme["primary"],
            fg="#0F172A" if self.theme_name == "midnight" else "#FFFFFF",
            hover_bg=self.theme["primary_hover"],
            active_bg=self.theme["primary_light"],
            font=("SF Pro Text", 10, "bold"),
            padx=14,
            pady=6,
            command=save,
            cursor="pointinghand"
        )
        btn.pack(pady=14)

    def import_context_dialog(self):
        """Opens file dialog and ingests ChatGPT export into active profile."""
        filepath = filedialog.askopenfilename(
            title="Import ChatGPT History (JSON, ZIP, or Markdown)",
            filetypes=[
                ("Supported Exports", "*.json *.zip *.md *.txt"),
                ("JSON exports", "*.json"),
                ("ZIP archives", "*.zip"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        if not filepath:
            return

        active_prof = self.selected_profile.get()
        parsed_entries = []

        try:
            if filepath.endswith('.zip'):
                with zipfile.ZipFile(filepath, 'r') as z:
                    for filename in z.namelist():
                        if filename.endswith('conversations.json'):
                            with z.open(filename) as f:
                                data = json.load(f)
                                parsed_entries.extend(self._parse_conversations_json(data))
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    parsed_entries.extend(self._parse_conversations_json(data))
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                    paras = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 30]
                    parsed_entries.extend(paras)

            if not parsed_entries:
                messagebox.showwarning("Import Warning", "No valid conversation turns found in the selected file.")
                return

            # Store in target profile ChromaDB collection
            safe_name = "prof_" + re.sub(r'[^a-zA-Z0-9_-]', '_', active_prof).lower()[:50]
            col = self.chroma_client.get_or_create_collection(name=safe_name)
            ids = [f"doc_{int(time.time()*1000)}_{i}" for i in range(len(parsed_entries))]
            col.add(documents=parsed_entries, ids=ids)

            self.load_profile_collection(active_prof)
            self.show_page("profiles")
            messagebox.showinfo("Context Ingested", f"Successfully ingested {len(parsed_entries)} dialogue chunks into profile [{active_prof}]!")

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to ingest file:\n{e}")

    def _parse_conversations_json(self, data) -> List[str]:
        parsed = []
        if not isinstance(data, list):
            data = [data]

        for chat in data:
            title = chat.get('title', 'Work Topic')
            mapping = chat.get('mapping', {})
            for _, node in mapping.items():
                msg = node.get('message')
                if msg and msg.get('content') and msg['content'].get('parts'):
                    role = msg.get('author', {}).get('role', 'user')
                    parts = msg['content']['parts']
                    t = " ".join([p for p in parts if isinstance(p, str)]).strip()
                    if t and len(t) > 15:
                        entry = f"[Thread: {title}] ({role}): {t}"
                        parsed.append(entry)
        return parsed

    def flush_context_confirm(self):
        """Wipes active profile ChromaDB collection after confirmation."""
        active = self.selected_profile.get()
        confirm = messagebox.askyesno(
            "Confirm Context Reset",
            f"Are you sure you want to completely erase all indexed knowledge fragments for [{active}]?\n\nThis action cannot be undone."
        )
        if confirm:
            try:
                safe_name = "prof_" + re.sub(r'[^a-zA-Z0-9_-]', '_', active).lower()[:50]
                self.chroma_client.delete_collection(name=safe_name)
                self.collection = self.chroma_client.get_or_create_collection(name=safe_name)
                self.show_page("profiles")
                messagebox.showinfo("Memory Reset", f"Profile [{active}] memory cleared successfully.")
            except Exception as e:
                messagebox.showerror("Reset Failed", f"Could not clear profile:\n{e}")

    # --- CONVERSATION AND MEETING RESUME EXPORT ---
    def open_export_dialog(self, source: str = "live"):
        """Ask whether to save the full conversation or the meeting resume, and in which format."""
        turns, _suggestions, _profile = self._export_payload(source)
        if not turns:
            messagebox.showinfo("Export", "There is no conversation to export yet.")
            return

        t = self.theme
        modal = tk.Toplevel(self.root)
        modal.title("Export call")
        modal.geometry("440x300")
        modal.configure(bg=t["surface"])
        modal.transient(self.root)
        modal.grab_set()

        tk.Label(
            modal,
            text="Save the full conversation or a meeting resume.",
            font=("SF Pro Text", 12, "bold"),
            fg=t["text_bright"],
            bg=t["surface"],
        ).pack(anchor="w", padx=18, pady=(16, 8))

        kind_var = tk.StringVar(value=KIND_CONVERSATION)
        fmt_var = tk.StringVar(value="md")
        radio = {
            "bg": t["surface"],
            "fg": t["text"],
            "selectcolor": t["surface_alt"],
            "activebackground": t["surface"],
            "activeforeground": t["text"],
            "font": ("SF Pro Text", 11),
            "highlightthickness": 0,
        }
        tk.Label(modal, text="What to export", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"]).pack(anchor="w", padx=18)
        tk.Radiobutton(modal, text="Full conversation", variable=kind_var, value=KIND_CONVERSATION, **radio).pack(anchor="w", padx=22)
        tk.Radiobutton(modal, text="Meeting resume", variable=kind_var, value=KIND_RESUME, **radio).pack(anchor="w", padx=22)
        tk.Label(modal, text="File format", font=("SF Pro Text", 10, "bold"), fg=t["muted"], bg=t["surface"]).pack(anchor="w", padx=18, pady=(8, 0))
        for key, label, _ext in FORMATS:
            tk.Radiobutton(modal, text=label, variable=fmt_var, value=key, **radio).pack(anchor="w", padx=22)

        actions = tk.Frame(modal, bg=t["surface"])
        actions.pack(fill="x", padx=18, pady=14)

        def close():
            try:
                modal.grab_release()
                modal.destroy()
            except Exception:
                pass

        def save():
            chosen_kind = kind_var.get()
            chosen_fmt = fmt_var.get()
            close()
            self._save_export(source, chosen_kind, chosen_fmt)

        ActionButton(
            actions, text="  Cancel", image=get_tk_icon("x", color=t["text"], size=14), compound="left",
            bg=t["surface_alt"], fg=t["text"], hover_bg=t["border"],
            active_bg=t["surface"], font=("SF Pro Text", 10, "bold"), padx=12, pady=6,
            command=close, cursor="pointinghand",
        ).pack(side="right", padx=(8, 0))
        ActionButton(
            actions, text="  Save file", image=get_tk_icon("save", color="#0F172A", size=14), compound="left",
            bg=t["primary"], fg="#0F172A", hover_bg=t["primary_hover"],
            active_bg=t["primary_light"], font=("SF Pro Text", 10, "bold"), padx=12, pady=6,
            command=save, cursor="pointinghand",
        ).pack(side="right")

    def _export_payload(self, source: str):
        profile = self.selected_profile.get()
        if source == "packet":
            path = os.path.join(REVIEWS_DIR, self.selected_packet_name or "")
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read()
            except Exception:
                return [], [], profile
            return turns_from_packet(content), [], profile_from_packet(content, profile)
        turns = list(self.transcript_history)
        if not turns and self.transcripts:
            turns = [
                (item.get("time", ""), item.get("speaker", ""), item.get("text", ""))
                for item in reversed(self.transcripts)
                if item.get("text")
            ]
        return turns, list(self.suggestion_history), profile

    def _save_export(self, source: str, kind: str, fmt: str):
        turns, suggestions, profile = self._export_payload(source)
        if not turns:
            messagebox.showinfo("Export", "There is no conversation to export yet.")
            return
        exports_dir = os.path.join(BASE_DIR, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = next(item[2] for item in FORMATS if item[0] == fmt)
        path = filedialog.asksaveasfilename(
            title="Export call",
            initialdir=exports_dir,
            initialfile=default_filename(kind, fmt, stamp),
            defaultextension=suffix,
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("CSV", "*.csv")],
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
            messagebox.showinfo("Export", f"Saved {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Export", f"Could not save the file:\n{exc}")

    # --- REVIEW PACKETS EXPORT ---
    def export_post_call_packet(self):
        """Generates offline review packet with ChatGPT sync prompt."""
        if not self.transcript_history:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        date_str = time.strftime("%B %d, %Y - %H:%M:%S")
        packet_filename = f"review_packet_{ts}.md"
        filepath = os.path.join(REVIEWS_DIR, packet_filename)
        active_prof = self.selected_profile.get()

        dialogue_lines = []
        for stamp, speaker, phrase in self.transcript_history:
            dialogue_lines.append(f"[{stamp}] {speaker}: {phrase}")
        transcript_body = "\n".join(dialogue_lines)

        fence = "```"
        packet_md = (
            f"# Live Call Post-Call Sync & Review Packet\n"
            f"**Recorded**: {date_str}\n"
            f"**Workspace Profile**: {active_prof}\n"
            f"**Total Dialogue Turns**: {len(self.transcript_history)}\n\n"
            f"---\n\n"
            f"## ChatGPT Workspace Memory Refresh Prompt\n"
            f"Copy and paste the entire block below into your ChatGPT workspace thread:\n\n"
            f"{fence}text\n"
            f"UPDATE_WORKSPACE_MEMORY_PROTOCOL_INIT\n"
            f"TARGET_PROFILE_CONTEXT: {active_prof}\n"
            f"CALL_TIMESTAMP: {date_str}\n\n"
            f"=== VERBATIM CALL TRANSCRIPT ===\n"
            f"{transcript_body}\n\n"
            f"INSTRUCTIONS FOR CHATGPT:\n"
            f"1. Extract newly established deliverables, approved budgets, adjusted deadlines, or commitments.\n"
            f"2. Flag any conflicts with historical context.\n"
            f"3. Generate an updated workspace memory profile update.\n"
            f"{fence}\n\n"
            f"---\n\n"
            f"## Captured Dialogue Timeline\n"
            f"{transcript_body}\n"
        )

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(packet_md)
            print(f"[Export System] Saved review packet to: {filepath}")
        except Exception as e:
            print(f"[Export Error]: {e}", file=sys.stderr)

    def refresh_review_packets(self):
        """Scans disk for packets."""
        files = [f for f in os.listdir(REVIEWS_DIR) if f.startswith("review_packet_") and f.endswith(".md")]
        files.sort(reverse=True)
        self.review_packets_list = files
        if files and not self.selected_packet_name:
            self.selected_packet_name = files[0]

    # --- GRACEFUL SHUTDOWN WORKFLOW WITH PROGRESS SPINNER ---
    def initiate_shutdown(self):
        """Displays shutdown modal with animated spinner and clean exit."""
        modal = tk.Toplevel(self.root)
        modal.title("Closing Application")
        modal.geometry("380x190")
        modal.configure(bg=self.theme["surface"])
        modal.transient(self.root)
        modal.grab_set()

        power_icon = get_tk_icon("power", color=self.theme["primary"], size=32)
        lbl_power = tk.Label(modal, image=power_icon, bg=self.theme["surface"])
        lbl_power.image = power_icon
        lbl_power.pack(pady=(16, 4))

        lbl_title = tk.Label(modal, text="Shutting Down Application...", font=("SF Pro Text", 12, "bold"), fg=self.theme["text_bright"], bg=self.theme["surface"])
        lbl_title.pack(pady=(0, 4))

        lbl_status = tk.Label(modal, text="Stopping live audio stream...", font=("SF Pro Text", 10), fg=self.theme["muted"], bg=self.theme["surface"])
        lbl_status.pack(pady=(0, 6))

        spinner_frames = ["|", "/", "-", "\\"]
        spin_idx = 0
        is_closing = True

        lbl_spin = tk.Label(modal, text="[ | ]", font=("SF Mono", 10, "bold"), fg=self.theme["primary"], bg=self.theme["surface"])
        lbl_spin.pack(pady=(0, 8))

        def animate():
            nonlocal spin_idx
            if is_closing and modal.winfo_exists():
                lbl_spin.config(text=f"[ {spinner_frames[spin_idx % len(spinner_frames)]} ]")
                spin_idx += 1
                modal.after(100, animate)

        animate()

        def set_status(text):
            if modal.winfo_exists():
                lbl_status.config(text=text)

        def finalize_and_exit():
            nonlocal is_closing
            is_closing = False
            try:
                modal.grab_release()
                modal.destroy()
                self.root.destroy()
            except Exception:
                pass
            os._exit(0)

        def perform_exit():
            # Step 1: Stop audio stream & export post call review packet
            if self.is_listening:
                self.is_listening = False
                self._close_audio_streams()
                try:
                    self.export_post_call_packet()
                except Exception:
                    pass

            self.safe_after(0, set_status, "Saving application configuration...")
            time.sleep(0.3)

            # Step 2: Persist config
            try:
                self.save_config()
            except Exception:
                pass

            self.safe_after(0, set_status, "Releasing audio and database resources...")
            time.sleep(0.3)

            self.safe_after(0, set_status, "Goodbye!")
            time.sleep(0.2)

            self.safe_after(0, finalize_and_exit)

        threading.Thread(target=perform_exit, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeAssistantApp(root)
    root.mainloop()
