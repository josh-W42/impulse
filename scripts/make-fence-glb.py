"""Generates assets/models/fence.glb: a chain-link fence panel (alpha-masked,
double-sided wire texture) between two opaque posts. Ours, CC0. Standard
library only.

Run from the repo root:  python scripts/make-fence-glb.py
"""

import json
import struct
import sys
import zlib
from pathlib import Path

WIDTH = 2.0  # panel width (m)
HEIGHT = 1.5  # panel height (m)
POST = 0.06  # post half-thickness (m)
TEXTURE = 64  # texture size (px); one diamond cell per texture
CELLS = (8.0, 6.0)  # texture repeats across the panel


def png(size, pixel):
    rows = b"".join(
        b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(size)) for y in range(size)
    )

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def wire(x, y):
    """A diamond lattice: two diagonal wires per cell, about 5 px wide."""
    s = TEXTURE
    u, v = (x + 0.5) / s, (y + 0.5) / s
    d1 = abs(((u + v) % 1.0) - 0.5) * s
    d2 = abs(((u - v) % 1.0) - 0.5) * s
    on = min(d1, d2) < 2.5
    shade = 150 + int(40 * (1.0 - min(d1, d2) / 2.5)) if on else 0
    return (shade, shade + 4, shade + 10, 255 if on else 0)


def box(cx, cz, hx, hy, hz, y0):
    """An axis-aligned box from y0 up: positions, normals, uvs, indices."""
    faces = [
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    ]
    half = (hx, hy, hz)
    centre = (cx, y0 + hy, cz)
    pos, nor, uv, idx = [], [], [], []
    for n, u, v in faces:
        base = len(pos)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = [centre[i] + half[i] * (n[i] + su * u[i] + sv * v[i]) for i in range(3)]
            pos.append(p)
            nor.append(list(n))
            uv.append([0.0, 0.0])
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return pos, nor, uv, idx


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/models/fence.glb")
    w, h = WIDTH / 2.0, HEIGHT
    panel = (
        [[-w, 0.0, 0.0], [w, 0.0, 0.0], [w, h, 0.0], [-w, h, 0.0]],
        [[0.0, 0.0, 1.0]] * 4,
        [[0.0, CELLS[1]], [CELLS[0], CELLS[1]], [CELLS[0], 0.0], [0.0, 0.0]],
        [0, 1, 2, 0, 2, 3],
    )
    posts = [[], [], [], []]
    for cx in (-w - POST, w + POST):
        p, n, t, i = box(cx, 0.0, POST, (h + 0.1) / 2.0, POST, 0.0)
        base = len(posts[0])
        posts[0] += p
        posts[1] += n
        posts[2] += t
        posts[3] += [base + k for k in i]

    binary = bytearray()
    views, accessors = [], []

    def add_view(data, target=None):
        while len(binary) % 4:
            binary.append(0)
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
        if target:
            view["target"] = target
        binary.extend(data)
        views.append(view)
        return len(views) - 1

    def add_floats(rows, kind, bounds=False):
        flat = [c for r in rows for c in r]
        view = add_view(struct.pack(f"<{len(flat)}f", *flat), 34962)
        acc = {"bufferView": view, "componentType": 5126, "count": len(rows), "type": kind}
        if bounds:
            acc["min"] = [min(r[i] for r in rows) for i in range(len(rows[0]))]
            acc["max"] = [max(r[i] for r in rows) for i in range(len(rows[0]))]
        accessors.append(acc)
        return len(accessors) - 1

    def add_indices(ids):
        view = add_view(struct.pack(f"<{len(ids)}H", *ids), 34963)
        accessors.append(
            {"bufferView": view, "componentType": 5123, "count": len(ids), "type": "SCALAR"}
        )
        return len(accessors) - 1

    def primitive(mesh, material):
        pos, nor, uv, idx = mesh
        return {
            "attributes": {
                "POSITION": add_floats(pos, "VEC3", bounds=True),
                "NORMAL": add_floats(nor, "VEC3"),
                "TEXCOORD_0": add_floats(uv, "VEC2"),
            },
            "indices": add_indices(idx),
            "material": material,
        }

    prims = [primitive(panel, 0), primitive(posts, 1)]
    image_view = add_view(png(TEXTURE, wire))
    while len(binary) % 4:
        binary.append(0)

    gltf = {
        "asset": {"version": "2.0", "generator": "impulse scripts/make-fence-glb.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "fence"}],
        "meshes": [{"primitives": prims}],
        "materials": [
            {
                "name": "chain-link",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.9,
                    "roughnessFactor": 0.45,
                },
                "alphaMode": "MASK",
                "alphaCutoff": 0.5,
                "doubleSided": True,
            },
            {
                "name": "post",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.35, 0.36, 0.38, 1.0],
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.5,
                },
            },
        ],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"wrapS": 10497, "wrapT": 10497}],
        "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = (
        struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(binary), 0x004E4942)
        + bytes(binary)
    )
    out.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    print(f"wrote {out} ({12 + len(body)} bytes)")


if __name__ == "__main__":
    main()
