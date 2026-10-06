"""Score the MILO teaser: narration (Kokoro-82M), sound effects and music,
all timed from the cues.json the teaser script wrote beside the video.

    python scripts/make_milo_audio.py --video outputs/trailers/milo_03/<stamp>/teaser.mp4 [--tag v1]

Narration: Kokoro-82M (kokoro-venv), voice am_michael (the deepest of the
voices tried), a touch slower, with a short trailer echo. A line that would
run into the next one is sped up to fit. Sound effects and music are
synthesised here with numpy (44.1 kHz). The music has a part per section of
the film (mystery, magic, danger, hero, title); it is ducked under the voice.

Writes to audio_<tag>/ beside the video, never over an existing file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE = 44100
# Python of the separate Kokoro environment (see scripts/kokoro_tts.py).
KOKORO_PY = Path(os.environ.get("KOKORO_PYTHON", Path.home() / "ai-tools/kokoro-venv/Scripts/python.exe"))
VOICE, SPEED = "am_michael", 0.92
rng = np.random.default_rng(3)


# --- building blocks -------------------------------------------------------------------

def t_axis(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * RATE)) / RATE


def note(name: str) -> float:
    names = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
    semis = names[name[0]] + (1 if "#" in name else 0) - (1 if "b" in name[1:-1] else 0) + 12 * (int(name[-1]) - 4)
    return 440.0 * 2 ** (semis / 12)


def saw(freq, t, detune=0.0):
    out = np.zeros_like(t)
    for d in (-detune, 0.0, detune):
        out += 2 * ((t * freq * (1 + d)) % 1.0) - 1
    return out / 3


def lowpass(x: np.ndarray, cutoff) -> np.ndarray:
    """One-pole lowpass; cutoff may be an array (a sweep)."""
    cutoff = np.broadcast_to(np.asarray(cutoff, float), x.shape)
    alpha = 1 - np.exp(-2 * np.pi * cutoff / RATE)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += alpha[i] * (x[i] - acc)
        y[i] = acc
    return y


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t * decay)


def noise(seconds: float) -> np.ndarray:
    return rng.uniform(-1, 1, int(seconds * RATE))


def sine_sweep(f0: float, f1: float, seconds: float, curve: float = 1.0) -> np.ndarray:
    t = t_axis(seconds)
    f = f0 + (f1 - f0) * (t / seconds) ** curve
    return np.sin(2 * np.pi * np.cumsum(f) / RATE)


# --- sound effects ---------------------------------------------------------------------

def sfx(name: str) -> np.ndarray:
    if name == "twinkle":
        out = np.zeros(int(0.9 * RATE))
        for k, n in enumerate(["E6", "G6", "B6", "E7"]):
            tt = t_axis(0.5)
            s = np.sin(2 * np.pi * note(n) * tt) * np.exp(-tt * 9) * 0.6
            a = int(k * 0.09 * RATE)
            out[a:a + len(s)] += s[: len(out) - a]
        return out
    if name in ("whoosh", "whoosh_up"):
        n = noise(0.6)
        sweep = np.linspace(300, 3000, len(n)) if name == "whoosh_up" else 2400 - 1800 * np.abs(np.linspace(-1, 1, len(n)))
        shape = np.sin(np.pi * np.linspace(0, 1, len(n))) ** 2
        return lowpass(n, sweep) * shape * 1.6
    if name == "boing":
        t = t_axis(0.7)
        f = 180 + 110 * np.sin(2 * np.pi * 11 * t) * np.exp(-5 * t)
        return np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-4 * t) * 0.7
    if name == "boing_take":
        t = t_axis(0.5)
        f = 350 + 700 * (t / 0.5) ** 0.5 + 60 * np.sin(2 * np.pi * 28 * t)
        return np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-3 * t) * 0.6
    if name == "chime":
        out = np.zeros(int(1.6 * RATE))
        for k, n in enumerate(["C6", "E6", "G6", "C7"]):
            tt = t_axis(1.4)
            s = (np.sin(2 * np.pi * note(n) * tt) + 0.4 * np.sin(2 * np.pi * note(n) * 2.76 * tt)) * np.exp(-tt * 3) * 0.35
            a = int(k * 0.07 * RATE)
            out[a:a + len(s)] += s[: len(out) - a]
        return out
    if name == "sting":
        t = t_axis(1.8)
        chord = sum(saw(note(n), t, 0.006) for n in ("A1", "Bb1", "E2", "F3"))
        hit = lowpass(chord, 900 * np.exp(-t * 1.5) + 150) * np.exp(-t * 1.6)
        burst = lowpass(noise(1.8), 1500) * np.exp(-t * 10) * 0.6
        return (hit + burst) * 0.9
    if name == "uh_oh":
        a, b = sine_sweep(note("G5"), note("G5"), 0.22), sine_sweep(note("D5"), note("C#5"), 0.35)
        out = np.concatenate([a * env(len(a), 0.01, 3), np.zeros(int(0.04 * RATE)), b * env(len(b), 0.01, 4)])
        return out * 0.35
    if name == "zip":
        whistle = sine_sweep(1800, 250, 0.3, 0.7) * np.exp(-t_axis(0.3) * 4)
        n = lowpass(noise(0.3), np.linspace(4000, 600, int(0.3 * RATE))) * np.exp(-t_axis(0.3) * 5)
        return (0.45 * whistle + 1.2 * n)
    if name == "grab":
        t = t_axis(0.4)
        thump = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 14)
        click = lowpass(noise(0.4), 2500) * np.exp(-t * 60)
        return (thump + 0.5 * click) * 0.9
    if name == "pop":
        s = sine_sweep(1100, 250, 0.09, 0.5) * np.exp(-t_axis(0.09) * 25)
        return np.concatenate([s, np.zeros(int(0.05 * RATE))]) * 0.6
    if name == "owl":  # two soft hoots, far away
        out = np.zeros(int(1.4 * RATE))
        for at, length in ((0.0, 0.35), (0.5, 0.6)):
            tt = t_axis(length)
            f = 390 * (1 - 0.06 * tt / length) + 6 * np.sin(2 * np.pi * 7 * tt)
            hoot = np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.sin(np.pi * tt / length) ** 2
            a = int(at * RATE)
            out[a:a + len(hoot)] += hoot
        return lowpass(out, 1200) * 0.35
    if name == "ding":  # a bright little bell for a wink
        tt = t_axis(1.0)
        return (np.sin(2 * np.pi * note("E6") * tt) + 0.5 * np.sin(2 * np.pi * note("B6") * tt)) * np.exp(-tt * 4) * 0.4
    if name == "loop":  # a slide whistle up and over
        tt = t_axis(0.9)
        f = 500 + 900 * np.sin(np.pi * tt / 0.9)
        return np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.sin(np.pi * tt / 0.9) * 0.3
    if name == "iris":  # a closing "whoop"
        return sine_sweep(900, 150, 0.6, 0.6) * np.exp(-t_axis(0.6) * 2) * 0.35
    raise ValueError(f"unknown sound effect {name!r}")


def ambience(kind: str, seconds: float) -> np.ndarray:
    t = t_axis(seconds)
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.4)
    if kind == "wind":  # low, gusting
        gust = 0.6 + 0.4 * np.sin(2 * np.pi * t / 3.1) * np.sin(2 * np.pi * t / 1.7)
        return lowpass(noise(seconds), 350 + 250 * gust) * gust * fade * 0.5
    if kind == "crickets":  # short high chirps in groups
        out = np.zeros_like(t)
        for start in np.arange(0.1, seconds - 0.3, 0.55):
            for k in range(3):
                tt = t_axis(0.03)
                a = int((start + k * 0.05 + rng.uniform(0, 0.02)) * RATE)
                chirp = np.sin(2 * np.pi * 4300 * tt) * np.sin(np.pi * tt / 0.03)
                out[a:a + len(chirp)] += chirp[: len(out) - a]
        return out * fade * 0.06
    raise ValueError(kind)


# --- music -----------------------------------------------------------------------------

def part_mystery(seconds: float) -> np.ndarray:
    t = t_axis(seconds)
    drone = (saw(note("A2"), t, 0.004) + 0.6 * saw(note("E3"), t, 0.003)) * np.minimum(1, t / 1.5)
    out = 0.25 * lowpass(drone, 500 + 200 * np.sin(2 * np.pi * t / 4))
    for i, n in enumerate((["A3", "C4", "E4", "G4", "E4", "C4"] * 6)[: int((seconds - 0.4) / 0.45)]):
        tt = t_axis(0.5)
        tone = lowpass(np.sign(np.sin(2 * np.pi * note(n) * tt)) * np.exp(-tt * 5), 1800)
        a = int((0.4 + i * 0.45) * RATE)
        out[a:a + len(tone)] += 0.06 * tone[: len(out) - a]
    return out


def part_magic(seconds: float) -> np.ndarray:
    t = t_axis(seconds)
    pad = sum(saw(note(n), t, 0.003) for n in ("C3", "G3", "E4")) * np.minimum(1, t / 0.6)
    out = 0.15 * lowpass(pad, 1000)
    for i, n in enumerate((["C5", "E5", "G5", "C6", "G5", "E5"] * 10)[: int(seconds / 0.2)]):
        tt = t_axis(0.22)
        a = int(i * 0.2 * RATE)
        s = np.sin(2 * np.pi * note(n) * tt) * np.exp(-tt * 10) * 0.08
        out[a:a + len(s)] += s[: len(out) - a]
    bass = np.zeros_like(t)
    for i, n in enumerate((["C2", "C2", "A1", "A1", "F1", "F1", "G1", "G1"] * 4)[: int(seconds / 0.4)]):
        tt = t_axis(0.38)
        s = lowpass(saw(note(n), tt), 600) * env(len(tt), 0.01, 4)
        a = int(i * 0.4 * RATE)
        bass[a:a + len(s)] += s[: len(bass) - a]
    return out + 0.18 * bass


def drums(seconds: float, bpm: float) -> np.ndarray:
    out = np.zeros(int(seconds * RATE))
    beat = 60 / bpm
    for i in range(int(seconds / (beat / 2))):
        a = int(i * beat / 2 * RATE)
        tt = t_axis(0.3)
        if i % 2 == 0:  # kick on every beat
            kick = np.sin(2 * np.pi * np.cumsum(50 + 90 * np.exp(-tt * 30)) / RATE) * np.exp(-tt * 10)
            out[a:a + len(kick)] += 0.8 * kick[: len(out) - a]
        if i % 4 == 2:  # snare on 2 and 4
            snare = lowpass(noise(0.3), 3500) * np.exp(-tt * 18) + 0.3 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 20)
            out[a:a + len(snare)] += 0.5 * snare[: len(out) - a]
        hat = (noise(0.05) - lowpass(noise(0.05), 6000)) * np.exp(-t_axis(0.05) * 70)
        out[a:a + len(hat)] += 0.25 * hat[: len(out) - a]
    return out


def part_danger(seconds: float) -> np.ndarray:
    t = t_axis(seconds)
    bass = np.zeros_like(t)
    step = 60 / 132 / 2
    for i, n in enumerate((["A1", "A1", "A2", "A1", "C2", "A1", "Bb1", "E2"] * 8)[: int(seconds / step)]):
        tt = t_axis(step * 0.95)
        s = lowpass(saw(note(n), tt, 0.004), 900) * env(len(tt), 0.005, 6)
        a = int(i * step * RATE)
        bass[a:a + len(s)] += s[: len(bass) - a]
    pad = lowpass(sum(saw(note(n), t, 0.005) for n in ("A3", "C4", "E4")), 700) * np.minimum(1, t / 0.3)
    return 0.32 * bass + 0.08 * pad + drums(seconds, 132)


def part_hero(seconds: float) -> np.ndarray:
    t = t_axis(seconds)
    chord = sum(saw(note(n), t, 0.004) for n in ("D3", "A3", "D4", "F#4")) * np.minimum(1, t / 0.15)
    out = 0.16 * lowpass(chord, 1800)
    for i, n in enumerate(["D5", "F#5", "A5", "D6", "F#6", "A6"][: int(seconds / 0.15)]):
        tt = t_axis(0.3)
        a = int(i * 0.15 * RATE)
        s = np.sin(2 * np.pi * note(n) * tt) * np.exp(-tt * 6) * 0.09
        out[a:a + len(s)] += s[: len(out) - a]
    return out + 0.6 * drums(seconds, 132)


def part_title(seconds: float) -> np.ndarray:
    t = t_axis(seconds)
    chord = sum(saw(note(n), t, 0.004) for n in ("C3", "E3", "G3", "C4", "E4")) * np.minimum(1, t / 0.3)
    out = 0.16 * lowpass(chord, 1400) * np.exp(-t * 0.4)
    tt = t_axis(2.5)
    s = np.sin(2 * np.pi * note("G6") * tt) * np.exp(-tt * 2) * 0.12
    out[: len(s)] += s[: len(out)]
    return out


def music(sections: dict, duration: float) -> np.ndarray:
    parts = {"mystery": part_mystery, "magic": part_magic, "danger": part_danger, "hero": part_hero, "title": part_title}
    out = np.zeros(int(duration * RATE) + RATE)
    fade = int(0.15 * RATE)
    for name, (a, b) in sections.items():
        x = parts[name](b - a + 0.15)
        x[:fade] *= np.linspace(0, 1, fade)
        x[-fade:] *= np.linspace(1, 0, fade)
        i = int(a * RATE)
        out[i:i + len(x)] += x[: len(out) - i]
    out = out[: int(duration * RATE)]
    tail = int(0.8 * RATE)
    out[-tail:] *= np.linspace(1, 0, tail)
    return out


# --- output ----------------------------------------------------------------------------

def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> None:
    data = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with open(path, "xb") as f, wave.open(f, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def read_wav(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float64) / 32768


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    video = Path(args.video).resolve()
    cues = json.loads((video.parent / "cues.json").read_text(encoding="utf-8"))
    folder = video.parent / f"audio_{args.tag}"
    folder.mkdir()
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     str(video)], capture_output=True, text=True, check=True).stdout)
    total = int(duration * RATE)

    # narration
    lines = cues["narration"]
    subprocess.run([str(KOKORO_PY), "-I", str(ROOT / "scripts/kokoro_tts.py"), str(folder), VOICE, str(SPEED),
                    *[text for _, text in lines]], check=True, capture_output=True)
    voice = np.zeros(total)
    report = []
    for n, (start, text) in enumerate(lines):
        x = read_wav(folder / f"line_{n}.wav")
        limit = (lines[n + 1][0] if n + 1 < len(lines) else duration - 0.3) - start - 0.1
        length = len(x) / RATE
        if length > limit:  # speed up to fit, keep the pitch
            fitted = folder / f"line_{n}_fit.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-n", "-i", str(folder / f"line_{n}.wav"), "-af",
                            f"atempo={min(length / limit, 1.6):.3f}", str(fitted)], check=True)
            x = read_wav(fitted)
        i = int(start * RATE)
        voice[i:i + len(x)] += x[: total - i]
        report.append({"start": start, "seconds": round(len(x) / RATE, 2), "limit": round(limit, 2), "text": text})
    write_wav(folder / "voice_dry.wav", voice)
    # a short trailer echo on the voice
    subprocess.run(["ffmpeg", "-loglevel", "error", "-n", "-i", str(folder / "voice_dry.wav"), "-af",
                    "aecho=0.85:0.6:45|90:0.25|0.12,acompressor=threshold=0.1:ratio=3,volume=1.5",
                    str(folder / "voice.wav")], check=True)
    voice = read_wav(folder / "voice.wav")[:total]
    voice = np.pad(voice, (0, total - len(voice)))

    effects = np.zeros(total)
    for at, name in cues["sfx"]:
        x = sfx(name)
        i = int(at * RATE)
        effects[i:i + len(x)] += x[: total - i]
    for kind, spans in cues.get("ambience", {}).items():
        for a, b in spans:
            x = ambience(kind, b - a)
            i = int(a * RATE)
            effects[i:i + len(x)] += x[: total - i]
    write_wav(folder / "sfx.wav", effects * 0.7)

    bed = music(cues["music"], duration)
    write_wav(folder / "music.wav", bed)

    # duck the music under the voice (envelope follower), then mix and limit
    level = np.convolve(np.abs(voice), np.ones(2205) / 2205, mode="same")
    duck = 1 - 0.6 * np.clip(level / 0.05, 0, 1)
    duck = np.convolve(duck, np.ones(4410) / 4410, mode="same")
    mix = 0.75 * bed * duck + 0.7 * effects + voice
    peak = np.abs(mix).max()
    mix = mix / peak * 0.89 if peak > 0.89 else mix
    write_wav(folder / "mix.wav", mix)
    final = folder / "teaser_with_audio.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-n", "-i", str(video), "-i", str(folder / "mix.wav"), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-map", "0:v", "-map", "1:a",
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(RATE), str(final)], check=True)
    (folder / "narration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in report:
        print(f"{r['start']:5.1f}s  {r['seconds']:4.2f}s (room {r['limit']:4.2f}s)  {r['text']}")
    print(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
