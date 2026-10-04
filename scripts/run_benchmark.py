"""Run one benchmark from the command line. See ai_studio.cli.

    python scripts/run_benchmark.py --benchmark comic_rooftop_01 --backend wan-ti2v-5b
        --seed 42 --width 1280 --height 704 [--param key=value ...] [--notes TEXT]
"""

import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
