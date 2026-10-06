"""Speak lines with Kokoro-82M. Run with the Python of the Kokoro environment
(by default ~/ai-tools/kokoro-venv; scripts/make_milo_audio.py reads the
KOKORO_PYTHON environment variable), not the project's:

    kokoro-venv/Scripts/python.exe -I scripts/kokoro_tts.py OUT_DIR VOICE SPEED "line one" "line two" ...

Writes OUT_DIR/line_<n>.wav (24 kHz mono), one file per line.
"""

import sys

import numpy as np
import soundfile as sf
from kokoro import KPipeline


def main() -> int:
    out_dir, voice, speed, *lines = sys.argv[1:]
    pipe = KPipeline(lang_code="a" if voice[0] == "a" else "b")
    for n, text in enumerate(lines):
        chunks = [np.asarray(audio) for _, _, audio in pipe(text, voice=voice, speed=float(speed))]
        sf.write(f"{out_dir}/line_{n}.wav", np.concatenate(chunks), 24000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
