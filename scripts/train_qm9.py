#!/usr/bin/env python3
"""Run QM9 training from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def main() -> None:
    """Load the source package and delegate to its training entrypoint."""
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from hympnn.train_qm9 import main as train

    train()


if __name__ == "__main__":
    main()
