"""Tiny CLI: `python -m app.frontend.cli oss` or `python -m app.frontend.cli frontier`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.backend import FrontierAssistant, OSSAssistant


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "oss"
    a = OSSAssistant() if backend == "oss" else FrontierAssistant()
    print(f"[{backend}] chat — Ctrl-D to exit\n")
    try:
        while True:
            msg = input("you> ").strip()
            if not msg:
                continue
            out = a.chat(msg)
            print(f"bot> {out['reply']}\n")
    except (EOFError, KeyboardInterrupt):
        print("\nbye")


if __name__ == "__main__":
    main()
