"""Flask entry point for the symmetry meme generator."""

from __future__ import annotations

import os
import re
import socket
import time
import uuid
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, abort, jsonify, render_template, request, send_file

from mirror import DIRECTION_LABELS, DIRECTIONS, MirrorError, flip, probe

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
FILE_TTL_SECONDS = 6 * 3600  # temp files older than this get swept away

HOST = os.environ.get("EMF_HOST", "127.0.0.1")
PORT = int(os.environ.get("EMF_PORT", "5000"))

ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# --------------------------------------------------------------------------
# Storage helpers
# --------------------------------------------------------------------------

def _sweep_old_files() -> None:
    """Delete expired uploads and outputs so the folders don't grow forever."""
    cutoff = time.time() - FILE_TTL_SECONDS
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        if not folder.is_dir():
            continue
        for item in folder.iterdir():
            try:
                if item.is_file() and item.stat().st_mtime < cutoff:
                    item.unlink()
            except OSError:
                pass  # locked by something else, try again next time


def _check_id(file_id: str) -> str:
    """Only accept uuid4 hex ids we handed out, which rules out path traversal."""
    if not ID_RE.match(file_id):
        abort(404)
    return file_id


def _find(folder: Path, stem: str) -> Path | None:
    for path in folder.glob(f"{stem}.*"):
        if path.is_file():
            return path
    return None


def _source_path(file_id: str) -> Path:
    path = _find(UPLOAD_DIR, _check_id(file_id))
    if path is None:
        abort(404)
    return path


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template(
        "index.html",
        directions=[(key, DIRECTION_LABELS[key]) for key in DIRECTIONS],
        max_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
    )


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.post("/api/upload")
def api_upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(ok=False, error="没有选择文件"), 400

    UPLOAD_DIR.mkdir(exist_ok=True)
    _sweep_old_files()

    # Stage to a temp file first, then sniff its contents. The stored name is
    # generated entirely by us; the client-supplied name is only ever echoed
    # back for display and never joined into a path. (The old code took the
    # extension straight off the uploaded filename and fed it to os.path.join,
    # which let a crafted name escape the uploads folder.)
    file_id = uuid.uuid4().hex
    staging = UPLOAD_DIR / f"{file_id}.part"
    file.save(staging)

    try:
        info = probe(staging)
    except MirrorError as exc:
        staging.unlink(missing_ok=True)
        return jsonify(ok=False, error=str(exc)), 400

    final = UPLOAD_DIR / f"{file_id}.{info.ext}"
    staging.replace(final)

    return jsonify(
        ok=True,
        id=file_id,
        name=file.filename,
        width=info.width,
        height=info.height,
        animated=info.animated,
        frames=info.n_frames,
        src_url=f"/media/src/{file_id}",
    )


@app.post("/api/flip")
def api_flip():
    payload = request.get_json(silent=True) or {}
    file_id = str(payload.get("id", ""))
    direction = str(payload.get("direction", ""))

    if direction not in DIRECTIONS:
        return jsonify(ok=False, error="方向不对"), 400

    src = _source_path(file_id)
    try:
        info = probe(src)
    except MirrorError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    OUTPUT_DIR.mkdir(exist_ok=True)
    dst = OUTPUT_DIR / f"{file_id}_{direction}.{info.ext}"

    if not dst.exists():
        try:
            flip(src, dst, direction, info.animated)
        except MirrorError as exc:
            return jsonify(ok=False, error=str(exc)), 400

    # Cache-buster so the browser doesn't show a stale result
    stamp = int(dst.stat().st_mtime)
    return jsonify(
        ok=True,
        url=f"/media/out/{file_id}/{direction}?v={stamp}",
        download_url=f"/download/{file_id}/{direction}",
    )


# --------------------------------------------------------------------------
# File serving
# --------------------------------------------------------------------------

@app.get("/media/src/<file_id>")
def media_src(file_id: str):
    return send_file(_source_path(file_id))


@app.get("/media/out/<file_id>/<direction>")
def media_out(file_id: str, direction: str):
    if direction not in DIRECTIONS:
        abort(404)
    path = _find(OUTPUT_DIR, f"{_check_id(file_id)}_{direction}")
    if path is None:
        abort(404)
    return send_file(path)


@app.get("/download/<file_id>/<direction>")
def download(file_id: str, direction: str):
    if direction not in DIRECTIONS:
        abort(404)
    path = _find(OUTPUT_DIR, f"{_check_id(file_id)}_{direction}")
    if path is None:
        abort(404)
    return send_file(
        path,
        as_attachment=True,
        download_name=f"对称_{DIRECTION_LABELS[direction]}{path.suffix}",
    )


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(_exc):
    return jsonify(ok=False, error=f"文件太大了，上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB"), 413


@app.errorhandler(404)
def not_found(_exc):
    if request.path.startswith(("/api/", "/media/", "/download/")):
        return jsonify(ok=False, error="找不到这个文件，可能已经过期了"), 404
    return render_template("index.html",
                           directions=[(k, DIRECTION_LABELS[k]) for k in DIRECTIONS],
                           max_mb=MAX_UPLOAD_BYTES // (1024 * 1024)), 404


@app.errorhandler(500)
def server_error(_exc):
    return jsonify(ok=False, error="服务器内部出错了，看一下控制台输出"), 500


def _port_taken(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def main() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    _sweep_old_files()

    url = f"http://{HOST}:{PORT}/"
    open_browser = os.environ.get("EMF_NO_BROWSER") != "1"

    # Double-clicking run.bat twice shouldn't blow up on a taken port; just
    # point the browser at the instance that is already serving.
    if _port_taken(HOST, PORT):
        print(f"\n  服务已经在运行了 -> {url}")
        print(f"  （如果想换端口：set EMF_PORT=5001 再启动）\n")
        if open_browser:
            webbrowser.open(url)
        return

    print(f"\n  对称表情包生成器已启动 -> {url}")
    print("  关掉这个窗口或按 Ctrl+C 停止\n")

    if open_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()

    # debug must stay off: the Werkzeug debugger allows arbitrary code
    # execution straight from the browser.
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
