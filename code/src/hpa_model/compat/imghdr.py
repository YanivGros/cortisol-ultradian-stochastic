"""Minimal Python 3.13 compatibility shim for packages that still import imghdr."""

from __future__ import annotations

from pathlib import Path


def what(file: str | bytes | Path | None, h: bytes | bytearray | None = None) -> str | None:
    if h is None:
        if file is None:
            return None
        path = Path(file)
        h = path.read_bytes()

    header = bytes(h[:32])

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if header.startswith(b"BM"):
        return "bmp"
    if header.startswith(b"RIFF") and b"WEBP" in header[:16]:
        return "webp"
    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "tiff"
    if header.startswith(b"\x00\x00\x01\x00"):
        return "ico"
    return None
