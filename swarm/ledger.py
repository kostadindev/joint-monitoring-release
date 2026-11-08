"""Persistent credit ledger for the swarm track (separate from Codex's ledgers).
Every paid call appends one JSON line; spending stops at CEILING across processes."""
import json, os, threading, time
from pathlib import Path

CEILING = 50.0
PATH = Path(__file__).resolve().parent / "out" / "ledger.jsonl"
_lock = threading.Lock()


def spent():
    if not PATH.exists():
        return 0.0
    return sum(json.loads(l).get("cost") or 0.0 for l in PATH.open())


class Ledger:
    def __init__(self, run_cap):
        self.run_cap, self.run_spent = run_cap, 0.0
        self.base = spent()

    def check(self):
        with _lock:
            if self.base + self.run_spent >= CEILING:
                raise RuntimeError(f"swarm ceiling {CEILING} credits reached")
            if self.run_spent >= self.run_cap:
                raise RuntimeError(f"run cap {self.run_cap} credits reached")

    def record(self, usage, tag):
        with _lock:
            self.run_spent += usage.get("cost") or 0.0
            PATH.parent.mkdir(parents=True, exist_ok=True)
            with PATH.open("a") as h:
                h.write(json.dumps({"t": time.time(), "tag": tag, **usage}) + "\n")
