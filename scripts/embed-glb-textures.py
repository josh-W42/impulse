#!/usr/bin/env python3
"""Embeds a .glb file's external image files into its binary chunk.

The engine loads only self-contained .glb files (a browser file picker can't
follow external references). Some asset packs ship .glb files that point at a
shared texture on disk; this rewrites each such image to a buffer view holding
the same bytes. Geometry, materials, and everything else are left untouched.

Usage: embed-glb-textures.py INPUT.glb OUTPUT.glb
Image URIs are resolved relative to INPUT's folder. Standard library only.
"""

import json
import struct
import sys
from pathlib import Path

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * (-len(data) % 4)


def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2 or length != len(data):
        raise SystemExit(f"{path}: not a glTF 2.0 binary")
    offset, doc, binary = 12, None, b""
    while offset < length:
        size, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + size]
        if kind == CHUNK_JSON:
            doc = json.loads(chunk)
        elif kind == CHUNK_BIN:
            binary = chunk
        offset += 8 + size
    if doc is None:
        raise SystemExit(f"{path}: no JSON chunk")
    return doc, binary


def embed(src: Path, dst: Path) -> int:
    doc, binary = read_glb(src)
    buffers = doc.setdefault("buffers", [])
    if not buffers:
        buffers.append({"byteLength": 0})
    if "uri" in buffers[0]:
        raise SystemExit(f"{src}: buffer 0 is external; not a self-contained .glb")
    views = doc.setdefault("bufferViews", [])
    embedded = 0
    for image in doc.get("images", []):
        uri = image.pop("uri", None)
        if uri is None:
            continue
        file = (src.parent / uri).resolve()
        mime = MIME.get(file.suffix.lower())
        if mime is None:
            raise SystemExit(f"{src}: unsupported image type {uri}")
        payload = file.read_bytes()
        binary = pad(binary, b"\0")
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        binary += payload
        image["bufferView"] = len(views) - 1
        image["mimeType"] = mime
        embedded += 1
    buffers[0]["byteLength"] = len(binary)
    body = pad(json.dumps(doc, separators=(",", ":")).encode(), b" ")
    blob = pad(binary, b"\0")
    total = 12 + 8 + len(body) + 8 + len(blob)
    out = struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(body), CHUNK_JSON) + body
    out += struct.pack("<II", len(blob), CHUNK_BIN) + blob
    dst.write_bytes(out)
    return embedded


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    count = embed(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{sys.argv[2]}: embedded {count} image(s)")
