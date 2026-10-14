import os
import sys
import time
import json
import struct
import socket
import signal
import atexit
import argparse
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

# Internal imports
from .searcher import GemmaSearchEngine, get_device

logger = logging.getLogger("gemmas.daemon")

def get_cache_dir() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache)
    else:
        base = Path.home() / ".cache"
    d = base / "gemmas"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_socket_path() -> Path:
    return get_cache_dir() / "daemon.sock"

def get_pid_path() -> Path:
    return get_cache_dir() / "daemon.pid"

def get_log_path() -> Path:
    return get_cache_dir() / "daemon.log"

# --- Socket IPC Framing ---

def send_msg(sock: socket.socket, data: Dict[str, Any]) -> None:
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError("Connection closed before reading complete message")
        buf.extend(chunk)
    return bytes(buf)

def recv_msg(sock: socket.socket) -> Optional[Dict[str, Any]]:
    try:
        header = recv_exact(sock, 4)
        if not header:
            return None
        (length,) = struct.unpack(">I", header)
        payload = recv_exact(sock, length)
        return json.loads(payload.decode("utf-8"))
    except (ConnectionResetError, EOFError):
        return None

# --- Daemon Implementation ---

class GemmasDaemon:
    def __init__(
        self,
        socket_path: Path,
        cooldown: int = 300,
        model_name: str = "google/embeddinggemma-2",
        device: str = "auto",
        dim: int = 256,
        batch_size: int = 32,
        max_seq_length: int = 512
    ):
        self.socket_path = socket_path
        self.pid_path = get_pid_path()
        self.cooldown = cooldown
        self.model_name = model_name
        self.requested_device = device
        self.dim = dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.start_time = time.time()
        self.last_activity = time.time()
        self.total_requests = 0
        self.stop_event = threading.Event()
        self.server_sock: Optional[socket.socket] = None

        logger.info(f"Loading Gemma model '{model_name}' on device '{device}'...")
        self.engine = GemmaSearchEngine(
            model_name=model_name,
            device=device,
            dim=dim,
            batch_size=batch_size,
            max_seq_length=max_seq_length
        )
        self.active_device = self.engine.device
        logger.info(f"Model successfully loaded on {self.active_device}.")

    def _cleanup(self) -> None:
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        if self.pid_path.exists():
            try:
                self.pid_path.unlink()
            except OSError:
                pass

    def _inactivity_watchdog(self) -> None:
        """Monitors idle time and triggers autonomous shutdown when cooldown is reached."""
        while not self.stop_event.is_set():
            time.sleep(1.0)
            idle_seconds = time.time() - self.last_activity
            if idle_seconds >= self.cooldown:
                logger.info(f"Cooldown limit of {self.cooldown}s reached without activity. Shutting down autonomously.")
                self.stop_event.set()
                # Connect to self to unblock accept()
                try:
                    dummy = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    dummy.connect(str(self.socket_path))
                    dummy.close()
                except OSError:
                    pass
                break

    def get_vram_info(self) -> Dict[str, Any]:
        info = {"device": self.active_device}
        if torch.cuda.is_available() and "cuda" in self.active_device:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            info["reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)
        return info

    def handle_client(self, conn: socket.socket) -> None:
        try:
            req = recv_msg(conn)
            if not req:
                return

            action = req.get("action")
            if action not in ("status", "ping"):
                self.last_activity = time.time()
            self.total_requests += 1

            if action == "ping":
                idle = round(time.time() - self.last_activity, 1)
                send_msg(conn, {
                    "status": "ok",
                    "uptime_seconds": round(time.time() - self.start_time, 1),
                    "idle_seconds": idle,
                    "cooldown": self.cooldown,
                    "vram": self.get_vram_info()
                })

            elif action == "status":
                idle = round(time.time() - self.last_activity, 1)
                send_msg(conn, {
                    "status": "running",
                    "pid": os.getpid(),
                    "model": self.model_name,
                    "device": self.active_device,
                    "uptime_seconds": round(time.time() - self.start_time, 1),
                    "idle_seconds": idle,
                    "cooldown": self.cooldown,
                    "total_requests": self.total_requests,
                    "vram": self.get_vram_info()
                })

            elif action == "stop":
                send_msg(conn, {"status": "stopping", "message": "Daemon shutting down."})
                self.stop_event.set()

            elif action == "search":
                query = req["query"]
                chunks = req["chunks"]
                mode = req.get("mode", "code")
                limit = req.get("limit", 5)
                dim = req.get("dim", self.dim)
                heat = req.get("heat", False)

                # Update engine dim if requested differently
                orig_dim = self.engine.dim
                self.engine.dim = dim

                # Pass 1: Coarse search
                ranked_results = self.engine.search(
                    query=query,
                    chunks=chunks,
                    mode=mode,
                    limit=limit,
                    show_progress=False
                )

                # Pass 2: Line saliency
                if heat and ranked_results:
                    top_chunks = [r[0] for r in ranked_results]
                    self.engine.compute_line_saliency(
                        query=query,
                        chunks=top_chunks,
                        mode=mode
                    )

                self.engine.dim = orig_dim

                # Format response payload
                matches = []
                for chunk, score in ranked_results:
                    matches.append({
                        "chunk": chunk,
                        "score": score
                    })

                send_msg(conn, {
                    "status": "ok",
                    "matches": matches,
                    "vram": self.get_vram_info()
                })

            else:
                send_msg(conn, {"status": "error", "message": f"Unknown action: {action}"})

        except Exception as e:
            logger.exception("Error handling IPC request")
            try:
                send_msg(conn, {"status": "error", "message": str(e)})
            except OSError:
                pass
        finally:
            conn.close()

    def run(self) -> None:
        self._cleanup()

        # Write PID
        with open(self.pid_path, "w") as f:
            f.write(str(os.getpid()))

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())
        signal.signal(signal.SIGINT, lambda s, f: self.stop_event.set())

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(str(self.socket_path))
        self.server_sock.listen(128)

        # Set permissions so only current user can connect
        os.chmod(self.socket_path, 0o600)

        # Start watchdog thread
        watchdog = threading.Thread(target=self._inactivity_watchdog, daemon=True)
        watchdog.start()

        logger.info(f"Gemmas daemon running (PID: {os.getpid()}, cooldown: {self.cooldown}s).")

        try:
            while not self.stop_event.is_set():
                try:
                    conn, _ = self.server_sock.accept()
                except OSError:
                    break

                if self.stop_event.is_set():
                    conn.close()
                    break

                self.handle_client(conn)
        finally:
            if self.server_sock:
                self.server_sock.close()
            self._cleanup()
            logger.info("Daemon shutdown complete.")

# --- Client Functions ---

def is_daemon_alive(socket_path: Optional[Path] = None) -> bool:
    sock_path = socket_path or get_socket_path()
    if not sock_path.exists():
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(str(sock_path))
        send_msg(s, {"action": "ping"})
        resp = recv_msg(s)
        s.close()
        return bool(resp and resp.get("status") == "ok")
    except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
        # Socket file exists but no process is listening
        try:
            sock_path.unlink()
        except OSError:
            pass
        return False

def spawn_daemon(
    cooldown: int = 300,
    model_name: str = "google/embeddinggemma-2",
    device: str = "auto",
    dim: int = 256,
    batch_size: int = 32,
    max_seq_length: int = 512
) -> None:
    sock_path = get_socket_path()
    log_path = get_log_path()

    cmd = [
        sys.executable,
        "-m",
        "gemmas.daemon",
        "--socket", str(sock_path),
        "--cooldown", str(cooldown),
        "--model", model_name,
        "--device", device,
        "--dim", str(dim),
        "--batch-size", str(batch_size),
        "--max-seq-len", str(max_seq_length),
    ]

    with open(log_path, "a") as log_file:
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )

def ensure_daemon(
    cooldown: int = 300,
    model_name: str = "google/embeddinggemma-2",
    device: str = "auto",
    dim: int = 256,
    batch_size: int = 32,
    max_seq_length: int = 512,
    timeout_sec: float = 35.0,
    verbose: bool = True
) -> bool:
    sock_path = get_socket_path()
    if is_daemon_alive(sock_path):
        return True

    if verbose:
        sys.stderr.write(f"[*] Starting gemmas daemon (cooldown: {cooldown}s) & loading Gemma into VRAM...\n")

    spawn_daemon(
        cooldown=cooldown,
        model_name=model_name,
        device=device,
        dim=dim,
        batch_size=batch_size,
        max_seq_length=max_seq_length
    )

    start = time.time()
    while time.time() - start < timeout_sec:
        if is_daemon_alive(sock_path):
            if verbose:
                sys.stderr.write("[*] Gemma daemon ready in memory.\n")
            return True
        time.sleep(0.3)

    sys.stderr.write("[-] Timed out waiting for gemmas daemon to start. See ~/.cache/gemmas/daemon.log\n")
    return False

def daemon_rpc(payload: Dict[str, Any], timeout: float = 60.0) -> Optional[Dict[str, Any]]:
    sock_path = get_socket_path()
    if not sock_path.exists():
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(sock_path))
        send_msg(s, payload)
        resp = recv_msg(s)
        s.close()
        return resp
    except Exception as e:
        logger.debug(f"RPC error: {e}")
        return None

# --- Entrypoint for python -m gemmas.daemon ---

def main() -> int:
    parser = argparse.ArgumentParser(description="Gemmas caching background daemon.")
    parser.add_argument("--socket", type=Path, default=get_socket_path())
    parser.add_argument("--cooldown", type=int, default=300, help="Inactivity cooldown in seconds (default: 300)")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    daemon = GemmasDaemon(
        socket_path=args.socket,
        cooldown=args.cooldown,
        model_name=args.model,
        device=args.device,
        dim=args.dim,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_len
    )
    daemon.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())
