"""Add narration and music to an assembled trailer (Windows; no model).

    python scripts/make_teaser_audio.py --video outputs/trailers/milo_02/<stamp>/trailer.mp4

Narration: Windows' built-in speech synthesiser (System.Speech, the only
installed voice), pitched down with ffmpeg for a deeper trailer voice; each
line starts at its time in LINES. Music: synthesised here with numpy, a
retro analog-synth cue in three parts that follow the cut (mysterious,
magical reveal, title), 44.1 kHz. The music is ducked under the voice.

Writes to audio_<tag>/ beside the video, never over an existing file:
    music.wav, narration_<n>_raw.wav, mix.wav, trailer_with_audio.mp4
The video stream is copied unchanged and keeps its full length (the audio is
padded or cut to it).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 44100
# (start second, text). milo_02's cut: forest 0-3, Milo appears 3-6, flies 6-10, waves 10-12, title 12-15.
LINES = [
    (0.4, "Deep in the shadowy forest..."),
    (3.3, "...lives a ghost who isn't scary at all."),
    (6.6, "Meet Milo!"),
    (11.6, "Milo. Coming soon!"),  # 3.3 s long: must end by 15 s
]
PITCH = 0.82  # voice pitched down to 82 %, tempo kept


def note(name: str) -> float:
    names = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
    semis = names[name[0]] + (1 if "#" in name else 0) + 12 * (int(name[-1]) - 4)
    return 440.0 * 2 ** (semis / 12)


def envelope(n: int, attack: float, release: float) -> np.ndarray:
    a, r = max(int(attack * RATE), 1), max(int(release * RATE), 1)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)[: n]
    env[-r:] *= np.linspace(1, 0, min(r, n))
    return env


def saw(freq: float, t: np.ndarray, detune: float = 0.0) -> np.ndarray:
    out = np.zeros_like(t)
    for d in (-detune, 0.0, detune):
        phase = (t * freq * (1 + d)) % 1.0
        out += 2 * phase - 1
    return out / 3


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    alpha = 1 - np.exp(-2 * np.pi * cutoff / RATE)
    y = np.zeros_like(x)
    acc = 0.0
    for i, v in enumerate(x):  # one-pole filter; 15 s of audio is fast enough
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def music(duration: float, reveal: float = 6.0, title: float = 12.0) -> np.ndarray:
    """Three parts: mysterious until `reveal`, magical until `title`, then the closing chord."""
    total = int(duration * RATE)
    mix = np.zeros(total)

    def add(start: float, length: float, signal: np.ndarray) -> None:
        a = int(start * RATE)
        b = min(a + len(signal), total)
        mix[a:b] += signal[: b - a]

    # Part 1 (0-6 s): dark A-minor drone and a slow arpeggio.
    t = np.arange(int((reveal + 0.3) * RATE)) / RATE
    drone = (saw(note("A2"), t, 0.004) + 0.6 * saw(note("E3"), t, 0.003)) * envelope(len(t), 1.5, 1.0)
    add(0.0, reveal + 0.3, 0.22 * lowpass(drone, 500))
    for i, n in enumerate((["A3", "C4", "E4", "G4", "E4", "C4"] * 4)[: max(1, int((reveal - 0.5) / 0.45))]):
        tt = np.arange(int(0.5 * RATE)) / RATE
        tone = np.sign(np.sin(2 * np.pi * note(n) * tt)) * np.exp(-tt * 5)
        add(0.5 + i * 0.45, 0.5, 0.05 * lowpass(tone, 1800))
    # Part 2 (6-12 s): the reveal - a bell chime, then a bright C-major arpeggio over a warm pad.
    for k, n in enumerate(["C6", "E6", "G6", "C7"]):
        tt = np.arange(int(1.6 * RATE)) / RATE
        bell = (np.sin(2 * np.pi * note(n) * tt) + 0.4 * np.sin(2 * np.pi * note(n) * 2.76 * tt)) * np.exp(-tt * 3)
        add(reveal - 0.1 + k * 0.09, 1.6, 0.10 * bell)
    t = np.arange(int((title - reveal + 0.3) * RATE)) / RATE
    pad = sum(saw(note(n), t, 0.003) for n in ("C3", "G3", "E4")) * envelope(len(t), 0.8, 1.2)
    add(reveal, title - reveal + 0.3, 0.13 * lowpass(pad, 900))
    for i, n in enumerate((["C5", "E5", "G5", "C6", "G5", "E5"] * 8)[: int((title - reveal - 0.3) / 0.24)]):
        tt = np.arange(int(0.25 * RATE)) / RATE
        tone = np.sin(2 * np.pi * note(n) * tt) * np.exp(-tt * 9)
        add(reveal + 0.2 + i * 0.24, 0.25, 0.07 * tone)
    # Part 3 (12-15 s): final chord swell and a last chime.
    t = np.arange(int((duration - title) * RATE)) / RATE
    chord = sum(saw(note(n), t, 0.004) for n in ("C3", "E3", "G3", "C4")) * envelope(len(t), 0.4, 1.6)
    add(title, duration - title, 0.16 * lowpass(chord, 1200))
    tt = np.arange(int(2.5 * RATE)) / RATE
    add(title + 0.1, 2.5, 0.12 * np.sin(2 * np.pi * note("G6") * tt) * np.exp(-tt * 2))

    mix *= np.r_[np.ones(total - int(0.8 * RATE)), np.linspace(1, 0, int(0.8 * RATE))]  # fade out
    return mix / max(np.abs(mix).max(), 1e-9) * 0.8


def write_wav(path: Path, samples: np.ndarray) -> None:
    data = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with open(path, "xb") as f, wave.open(f, "wb") as w:  # never over an existing file
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


def speak(text: str, path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = -1; "
        f"$s.SetOutputToWaveFile('{path}'); $s.Speak(\"{text}\"); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--tag", default="v1", help="output folder audio_<tag> beside the video")
    parser.add_argument("--no-narration", action="store_true", help="music only")
    parser.add_argument("--reveal", type=float, default=6.0, help="second the music turns magical")
    parser.add_argument("--title", type=float, default=12.0, help="second of the closing chord")
    args = parser.parse_args()
    video = Path(args.video).resolve()
    folder = video.parent / f"audio_{args.tag}"
    folder.mkdir(exist_ok=True)
    final = folder / "trailer_with_audio.mp4"
    if final.exists():
        print(f"ERROR: {final} exists")
        return 1
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout)

    write_wav(folder / "music.wav", music(duration, args.reveal, args.title))
    if args.no_narration:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-n", "-i", str(video), "-i", str(folder / "music.wav"),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v", "-map", "1:a",
                        "-af", "volume=0.9", str(final)], check=True)
        print(final)
        return 0
    inputs, filters = ["-i", str(folder / "music.wav")], []
    for k, (start, text) in enumerate(LINES):
        raw = folder / f"narration_{k}_raw.wav"
        speak(text, raw)
        inputs += ["-i", str(raw)]
        rate = 22050  # System.Speech writes 22.05 kHz by default
        filters.append(
            f"[{k + 1}:a]asetrate={rate * PITCH:.0f},aresample={RATE},atempo={1 / PITCH:.4f},"
            f"volume=1.2,adelay={int(start * 1000)}|{int(start * 1000)}[v{k}]"
        )
    voices = "".join(f"[v{k}]" for k in range(len(LINES)))
    filters.append(f"{voices}amix=inputs={len(LINES)}:normalize=0,apad,atrim=0:{duration}[voice]")
    filters.append("[voice]asplit[voice1][voice2]")
    filters.append("[0:a]volume=0.55[bed]")
    filters.append("[bed][voice1]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=300[ducked]")
    filters.append("[ducked][voice2]amix=inputs=2:normalize=0,volume=0.8,alimiter=limit=0.89[mix]")
    mix = folder / "mix.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-n", *inputs, "-filter_complex", ";".join(filters),
                    "-map", "[mix]", "-ar", str(RATE), str(mix)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-n", "-i", str(video), "-i", str(mix), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-map", "0:v", "-map", "1:a", str(final)], check=True)
    print(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
