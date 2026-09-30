"""Small media value types shared by generation and benchmarking."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"resolution {name} must be a positive int")


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# JPEG start-of-frame markers that carry dimensions (excludes DHT/JPG/DAC).
_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def image_dimensions(path: str | Path) -> Resolution | None:
    """Width/height read from a PNG or JPEG header, without decoding or
    modifying the file. None for other formats or unreadable headers."""
    with open(path, "rb") as f:
        head = f.read(26)
        if head[:8] == _PNG_SIGNATURE and head[12:16] == b"IHDR":
            width, height = struct.unpack(">II", head[16:24])
            return Resolution(width=width, height=height) if width and height else None
        if head[:2] != b"\xff\xd8":
            return None
        f.seek(2)
        while True:
            marker = f.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                return None
            if marker[1] in (0xD8, 0x01) or 0xD0 <= marker[1] <= 0xD7:
                continue  # markers without a length field
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                return None
            (length,) = struct.unpack(">H", length_bytes)
            if marker[1] in _JPEG_SOF:
                data = f.read(5)
                if len(data) < 5:
                    return None
                height, width = struct.unpack(">HH", data[1:5])
                return Resolution(width=width, height=height) if width and height else None
            f.seek(length - 2, 1)
