"""A minimal stand-in for ComfyUI's HTTP API, for tests. Standard library only.

Implements what the adapter uses: GET /system_stats, GET /object_info/<node>,
POST /upload/image, POST /prompt, GET /history/<id>, GET /view, POST /interrupt.
"Generated" frames are synthetic PNGs of the size and count the workflow asks
for.

Use in-process:      server = FakeComfyUI(); server.start(); ...; server.stop()
or as a program:     python main.py --listen 127.0.0.1 --port 8188
(the adapter starts ComfyUI as `python main.py ...`, so a copy of this file
named main.py behaves like a ComfyUI install).
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
import urllib.parse
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UNET = "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf"
BASE_UNET = "Wan2.2-TI2V-5B-Q8_0.gguf"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan2.2_vae.safetensors"

NODES = {
    "UnetLoaderGGUF": {"required": {"unet_name": [[UNET, BASE_UNET]]}},
    "CLIPLoader": {
        "required": {"clip_name": [[TEXT_ENCODER]], "type": [["stable_diffusion", "wan"]]},
        "optional": {"device": [["default", "cpu"], {"advanced": True}]},
    },
    "VAELoader": {"required": {"vae_name": [[VAE]]}},
    "LoadImage": {"required": {"image": [["example.png"], {"image_upload": True}]}},
    "CLIPTextEncode": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}},
    "ModelSamplingSD3": {"required": {"model": ["MODEL"], "shift": ["FLOAT", {"default": 3.0}]}},
    "Wan22ImageToVideoLatent": {
        "required": {"vae": ["VAE"], "width": ["INT", {}], "height": ["INT", {}], "length": ["INT", {}],
                     "batch_size": ["INT", {}]},
        "optional": {"start_image": ["IMAGE"]},
    },
    "KSampler": {"required": {
        "model": ["MODEL"], "seed": ["INT", {}], "steps": ["INT", {}], "cfg": ["FLOAT", {}],
        "sampler_name": [["euler", "uni_pc"]], "scheduler": ["COMBO", {"options": ["simple", "normal", "beta"]}],
        "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"],
        "denoise": ["FLOAT", {}],
    }},
    "LatentMultiply": {"required": {"samples": ["LATENT"], "multiplier": ["FLOAT", {}]}},
    "VAEDecodeTiled": {"required": {
        "samples": ["LATENT"], "vae": ["VAE"], "tile_size": ["INT", {}], "overlap": ["INT", {}],
        "temporal_size": ["INT", {}], "temporal_overlap": ["INT", {}],
    }},
    "SaveImage": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}},
}


def png_bytes(width: int, height: int, variant: int) -> bytes:
    """A blocky RGB test image; `variant` changes the pattern."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = []
    for block_y in range(height // 32 + 1):
        row = b"".join(
            bytes(((block_x * 37 + block_y * 11 + variant * 53) % 200 + 20,
                   (block_x * 17 + variant * 29) % 200 + 20,
                   (block_y * 23 + variant * 7) % 200 + 20)) * 32
            for block_x in range(width // 32 + 1)
        )[: width * 3]
        rows.append((b"\x00" + row) * 32)
    raw = b"".join(rows)[: height * (width * 3 + 1)]
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


class FakeComfyUI:
    def __init__(self, port: int = 0, *, missing_nodes=(), never_finish=False, fail_with=None,
                 frames=None, polls_until_done=2) -> None:
        self.missing_nodes = set(missing_nodes)
        self.never_finish = never_finish
        self.fail_with = fail_with  # error message: the prompt "fails" in ComfyUI
        self.frames = frames  # None: as many as the workflow asks for
        self.polls_until_done = polls_until_done
        self.prompts: dict[str, dict] = {}
        self.polls: dict[str, int] = {}
        self.uploads: list[str] = []
        self.interrupted = 0
        self.views = 0
        self._png_cache: dict[tuple[int, int, int], bytes] = {}
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep test output quiet
                pass

            def _send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, payload, status: int = 200) -> None:
                self._send(json.dumps(payload).encode(), status=status)

            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/system_stats":
                    self._json({"system": {"comfyui_version": "0.0-fake", "pytorch_version": "2.11.0+cu128"}})
                elif parsed.path.startswith("/object_info/"):
                    name = urllib.parse.unquote(parsed.path.split("/", 2)[2])
                    known = name in NODES and name not in owner.missing_nodes
                    self._json({name: {"input": NODES[name]}} if known else {})
                elif parsed.path.startswith("/history/"):
                    self._json(owner.history(parsed.path.split("/", 2)[2]))
                elif parsed.path == "/view":
                    query = urllib.parse.parse_qs(parsed.query)
                    self._send(owner.frame(query["filename"][0]), "image/png")
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/upload/image":
                    name = body.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
                    owner.uploads.append(name)
                    self._json({"name": name, "subfolder": "", "type": "input"})
                elif self.path == "/prompt":
                    workflow = json.loads(body)["prompt"]
                    unknown = sorted(
                        node["class_type"] for node in workflow.values()
                        if node["class_type"] not in NODES or node["class_type"] in owner.missing_nodes
                    )
                    if unknown:
                        self._json({"error": f"unknown node classes {unknown}"}, 400)
                        return
                    prompt_id = uuid.uuid4().hex
                    owner.prompts[prompt_id] = workflow
                    self._json({"prompt_id": prompt_id, "number": len(owner.prompts)})
                elif self.path == "/interrupt":
                    owner.interrupted += 1
                    self._json({})
                else:
                    self._json({"error": "not found"}, 404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.httpd.server_address[1]

    def history(self, prompt_id: str) -> dict:
        workflow = self.prompts.get(prompt_id)
        with self._lock:
            self.polls[prompt_id] = self.polls.get(prompt_id, 0) + 1
            polls = self.polls[prompt_id]
        if workflow is None or self.never_finish or polls < self.polls_until_done:
            return {}
        if self.fail_with:
            return {prompt_id: {"outputs": {}, "status": {
                "status_str": "error", "completed": False,
                "messages": [["execution_error", {
                    "node_type": "KSampler", "exception_type": "torch.OutOfMemoryError",
                    "exception_message": self.fail_with,
                }]],
            }}}
        latent = next(n for n in workflow.values() if n["class_type"] == "Wan22ImageToVideoLatent")["inputs"]
        save_id, save = next((i, n) for i, n in workflow.items() if n["class_type"] == "SaveImage")
        prefix = save["inputs"]["filename_prefix"]
        subfolder, _, stem = prefix.rpartition("/")
        count = self.frames if self.frames is not None else latent["length"]
        images = [
            {"filename": f"{stem}_{index + 1:05d}_{latent['width']}x{latent['height']}.png",
             "subfolder": subfolder, "type": "output"}
            for index in range(count)
        ]
        return {prompt_id: {"outputs": {save_id: {"images": images}},
                            "status": {"status_str": "success", "completed": True, "messages": []}}}

    def frame(self, filename: str) -> bytes:
        stem, size = filename[:-4].rsplit("_", 1)
        width, height = (int(v) for v in size.split("x"))
        variant = int(stem.rsplit("_", 1)[1]) % 4  # a few distinct frames, repeated
        key = (width, height, variant)
        with self._lock:
            self.views += 1
            if key not in self._png_cache:
                self._png_cache[key] = png_bytes(width, height, variant)
            return self._png_cache[key]

    def start(self) -> FakeComfyUI:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--missing-node", action="append", default=[])
    parser.add_argument("--exit-immediately", action="store_true")
    args, _ = parser.parse_known_args()  # tolerate ComfyUI flags such as --disable-auto-launch
    if args.exit_immediately:
        raise SystemExit("fake ComfyUI: refusing to start")
    print(f"fake ComfyUI listening on {args.listen}:{args.port}", flush=True)
    FakeComfyUI(args.port, missing_nodes=args.missing_node).httpd.serve_forever()
