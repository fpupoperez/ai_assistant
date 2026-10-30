"""Input devices plus macOS Multi-Output devices for the meeting selector.

A Multi-Output Device is output-only, so PortAudio hides it from input lists.
The call is still audible on the loopback member inside that device, usually
BlackHole, and that member is what the capture stream opens.
"""

import ctypes
from ctypes import byref, c_uint32, create_string_buffer
from typing import Dict, List, Optional, Tuple

import sounddevice as sd

MULTI_PREFIX = "multi:"
LABEL_LIMIT = 28


def short_device_label(name: str, limit: int = LABEL_LIMIT) -> str:
    """Keep a device name inside the meeting selector."""
    text = " ".join(str(name or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _fourcc(text: str) -> int:
    return int.from_bytes(text.encode("ascii"), "big")


def _input_index_by_name(name: str) -> Optional[int]:
    for index, dev in enumerate(sd.query_devices()):
        if dev["name"] == name and dev["max_input_channels"] > 0:
            return index
    return None


def multi_output_devices() -> List[Dict[str, object]]:
    """Return Multi-Output / aggregate devices and the input used to record them."""
    try:
        members = _aggregate_members()
    except Exception:
        return []

    listed = []
    for name, member_names in members:
        capture_index = None
        capture_name = None
        ordered = sorted(member_names, key=lambda item: (0 if "blackhole" in item.lower() else 1, item.lower()))
        for member in ordered:
            found = _input_index_by_name(member)
            if found is not None:
                capture_index = found
                capture_name = member
                break
        listed.append({
            "id": MULTI_PREFIX + name,
            "label": short_device_label(name),
            "capture_index": capture_index,
            "capture_name": capture_name,
        })
    return listed


def resolve_meeting_device(choice: str):
    """Map a meeting-selector value to a PortAudio input device."""
    choice = str(choice or "auto")
    if choice.startswith(MULTI_PREFIX):
        for device in multi_output_devices():
            if device["id"] == choice and device["capture_index"] is not None:
                index = int(device["capture_index"])
                return index, sd.query_devices(index)
        return None

    if choice not in ("", "auto"):
        try:
            index = int(choice)
            dev = sd.query_devices(index)
        except Exception:
            return None
        if dev["max_input_channels"] <= 0:
            return None
        return index, dev

    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
            return index, dev
    return None


def _aggregate_members() -> List[Tuple[str, List[str]]]:
    core = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    core.AudioObjectGetPropertyData.restype = c_uint32
    core.AudioObjectGetPropertyData.argtypes = [
        c_uint32,
        ctypes.c_void_p,
        c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(c_uint32),
        ctypes.c_void_p,
    ]

    class Address(ctypes.Structure):
        _fields_ = [("selector", c_uint32), ("scope", c_uint32), ("element", c_uint32)]

    scope = _fourcc("glob")

    def read_property(obj: int, selector: str, size_hint: int = 1024):
        address = Address(_fourcc(selector), scope, 0)
        size = c_uint32(size_hint)
        buffer = create_string_buffer(size_hint)
        error = core.AudioObjectGetPropertyData(c_uint32(obj), byref(address), c_uint32(0), None, byref(size), buffer)
        if error != 0:
            return None
        return buffer.raw[: size.value]

    def device_name(device_id: int) -> str:
        raw = read_property(device_id, "name", 256)
        if not raw:
            return ""
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    raw_ids = read_property(1, "dev#", 4096)
    if not raw_ids:
        return []
    device_ids = [int.from_bytes(raw_ids[offset:offset + 4], "little") for offset in range(0, len(raw_ids), 4)]

    aggregates = []
    for device_id in device_ids:
        raw_members = read_property(device_id, "agrp", 256)
        if not raw_members:
            continue
        member_ids = [int.from_bytes(raw_members[offset:offset + 4], "little") for offset in range(0, len(raw_members), 4)]
        name = device_name(device_id)
        member_names = [device_name(member_id) for member_id in member_ids if device_name(member_id)]
        if name and member_names:
            aggregates.append((name, member_names))
    return aggregates
