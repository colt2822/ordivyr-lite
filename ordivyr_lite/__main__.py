"""``python -m ordivyr_lite`` and console-script entry point."""

from __future__ import annotations

import sys


def main() -> int:
    # Avoid importing the CLI graph for the one command every installer runs.
    if len(sys.argv) == 2 and sys.argv[1] in ("--version", "-V"):
        from . import __version__
        print(f"ordivyr-lite {__version__}")
        return 0
    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
