# Running a benchmark on Camber (from Windows)

Camber gives a free single NVIDIA L4 GPU node. This guide takes the project
from your Windows machine to Camber, runs `comic_rooftop_01` on
Wan 2.2 TI2V-5B there, and brings the result back.

> **Status: corrected after the first real session on Camber; no successful
> run yet.** Upload, unzip, setup, model loading and denoising worked on the
> L4. Both runs then failed in the VAE decode: first out of memory, then a
> tiling bug in diffusers 0.35.2. The project now pins diffusers 0.36.0 and
> enables VAE tiling by default. That combination has **not** run on Camber
> yet, so the decode step, the encoding of real frames and the total time are
> still unconfirmed. Do the smoke run (step 5) first.
>
> **Update:** a smoke run with exactly this setup has since succeeded
> (`20261004T145422Z-a264ad6e`: 12.4 min in total, of which 7.1 min loading and
> 5.3 min for 2 steps, the tiled decode and encoding). The full 50-step run
> has not been done yet, so its ~35 min is still an estimate.

## GPU budget: read this first

- The free GPU budget is **5 GPU-hours per month**.
- GPU time is counted **while the GPU node is on**, not while a run is
  executing. An idle GPU node still spends the budget.
- So: do everything you can on the **CPU node** (upload, unzip, setup), start
  the GPU node only for the runs, and **stop it as soon as the last run has
  finished**. Downloading the result does not need the GPU node.

Measured on the L4 (1280×704, 121 frames, `cpu_offload=true`):

| | |
|---|---|
| Loading the model | ~6.3 min per session (network disk; ~92 s per fp32 transformer shard) |
| Denoising | ~30 s per step |
| Full run, 50 steps | ~25 min of denoising + load + decode: **~35 min** in total (decode time with tiling is an estimate; it has not completed yet) |
| Smoke run, 2 steps | ~1 min of denoising + load + decode: roughly 8–10 min |

A session with one smoke run and one full run is therefore about 45 minutes
of the 300 minutes per month. Whether the second run in a session loads
faster than the first is not measured yet.

## Which node is which

The two node names are easy to confuse:

| Use it for | Camber name | Size |
|---|---|---|
| upload, unzip, setup, download | **Xxsmall** (CPU) | 4 CPU / 16 GB, free |
| the runs only | **GPU · Xsmall** | 8 CPU / 32 GB / 1 L4, counts against the 5 GPU-hours |

"Xxsmall" (two x) is the CPU node. "GPU · Xsmall" (one x, with "GPU" in
front) is the GPU node. Check the name before starting a node.

## What is where on Camber

| | |
|---|---|
| Persistent disk | `/home/jovyan` only (network disk, **50 GB quota**). Everything else, including `/tmp` and `/opt/conda`, is wiped when the node stops |
| Project | `/home/jovyan/ai-animation-studio` |
| Python environment | `/home/jovyan/envs/ai-studio` (conda, Python 3.11), made by `camber_setup.py` |
| Model weights | `/home/jovyan/hf` (`HF_HOME`), already downloaded: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` at revision `b8fff7315c768468a5333511427288870b2e9635`, 34.20 GB |
| Run zips | `/home/jovyan/ai-animation-studio/dist/run-<benchmark>-<run_id>.zip` |

Disk: the weights take 34.2 GB of the 50 GB. The environment adds several GB
more, so there is little room left. `camber_setup.py` prints the totals and
warns above 45 GB. Each run adds its video and a zip of it; delete old
`dist/*.zip` and `outputs/` folders on Camber once they are imported on
Windows.

**Notebook behaviour.** All cells below are plain Python. Do not use
`%%bash` cells: text pasted from Windows carries CRLF line endings, which
breaks them. A Camber notebook shows a cell's output **only when the cell has
finished**, so a run cell looks silent for its whole duration. That is
normal; see step 5 for how to check progress meanwhile.

## Step 1 (Windows): make the bundle

Commit your work first; the bundle must be exactly one commit.

```
python scripts\make_bundle.py
```

It prints the zip path and size: `dist\ai-animation-studio-bundle.zip`. The
bundle contains every git-tracked file, the `.git` directory (so the run
records the real `git_commit`) and the canonical start frame
`benchmarks/references/comic_hero_01/rooftop_frame_v001.png`, whose SHA-256 is
verified before packing. It refuses a working tree with uncommitted or
untracked files; `--allow-dirty` overrides that, and the run is then recorded
with `git_dirty: true`.

## Step 2 (Camber, CPU node "Xxsmall"): upload and unzip

Start the **Xxsmall** CPU node. Upload `ai-animation-studio-bundle.zip` with
the **Stash upload button**. The file lands in **the folder that is open in
the file browser** at that moment, not in a fixed place (the first time it
ended up in `/home/jovyan/projects/ai-animation-studio/`). The cell below
therefore searches for it.

If you are **updating** an existing copy, run this cleanup cell first. It
removes the old code so that files deleted from the project do not linger;
`benchmarks/results/`, `outputs/` and `dist/` are kept.

```python
import shutil
for name in ("src", "scripts", "tests", "docs", "requirements", ".git"):
    shutil.rmtree(f"/home/jovyan/ai-animation-studio/{name}", ignore_errors=True)
print("old code removed")
```

Unzip and verify:

```python
import glob, hashlib, os, subprocess, zipfile

NAME = "ai-animation-studio-bundle.zip"
PROJECT = "/home/jovyan/ai-animation-studio"
FRAME = PROJECT + "/benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
FRAME_SHA256 = "3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb"

found = [p for p in glob.glob("/home/jovyan/*.zip") + glob.glob("/home/jovyan/projects/*/*.zip")
         if os.path.basename(p) == NAME]
assert found, "bundle zip not found in /home/jovyan or /home/jovyan/projects/*/"
bundle = max(found, key=os.path.getmtime)  # the newest upload
print("bundle:", bundle, f"({os.path.getsize(bundle) / 1e6:.1f} MB)")
with zipfile.ZipFile(bundle) as z:
    z.extractall("/home/jovyan")

def git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True).stdout.strip()

print("commit:", git("log", "-1", "--format=%h %s"))
status = git("status", "--porcelain")
print("git status:", "clean" if not status else "NOT CLEAN\n" + status)
digest = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
print("start frame sha256:", "OK" if digest == FRAME_SHA256 else "WRONG " + digest)
```

Check the three lines it prints:

- `commit:` is the commit you bundled on Windows
- `git status: clean`. Lines starting with `??` under `benchmarks/results/`
  are records of earlier runs on Camber and are fine (see Known limits);
  anything else means the unzip did not replace the old code: run the cleanup
  cell and unzip again
- `start frame sha256: OK`

## Step 3 (Camber, CPU node "Xxsmall"): set up the environment

Needed the first time, and **again whenever `requirements/camber-wan.txt`
changes**. It changed with this version (diffusers 0.35.2 → 0.36.0), so run
it again after uploading this bundle. It is safe to repeat: the existing
conda environment is kept and pip only replaces what changed.

```python
import runpy
code = runpy.run_path("/home/jovyan/ai-animation-studio/scripts/camber_setup.py")["main"]()
print("exit code:", code)
```

(This is the same as `python3 scripts/camber_setup.py` with the base python.)

What it does:

1. creates the conda env `/home/jovyan/envs/ai-studio` with Python 3.11
   (from conda-forge) if it does not exist, with conda's package cache in
   `/tmp/conda-pkgs` and pip's cache in `/tmp/pip-cache` so no cache lands on
   the 50 GB quota
2. `pip install -r requirements/camber-wan.txt`
3. `pip install -e . --no-deps` (the project itself)
4. verifies `import torch, torchvision, diffusers, transformers, ai_studio`
   and prints the versions
5. prints the size of `/home/jovyan/envs`, `/home/jovyan/hf` and the whole
   home directory, and warns above 45 GB

It must end with `SETUP OK` and `exit code: 0`, and the version lines must
show **`diffusers 0.36.0`**. `cuda available False` is expected on the CPU
node. On any error it stops and prints `SETUP FAILED: ...`.

Pinned packages (`requirements/camber-wan.txt`, Python 3.11, Linux):

| Package | Version |
|---|---|
| torch | 2.8.0+cu126 (PyTorch index `https://download.pytorch.org/whl/cu126`) |
| torchvision | 0.23.0+cu126 |
| diffusers | **0.36.0** |
| transformers | 4.56.2 |
| tokenizers | 0.22.1 |
| safetensors | 0.6.2 |
| sentencepiece | 0.2.1 |
| accelerate | 1.10.1 |
| huggingface_hub | 0.35.3 |
| ftfy | 6.3.1 |
| pillow | 11.3.0 |
| numpy | 2.2.6 |
| pyyaml | 6.0.3 |

Why diffusers 0.36.0: in 0.35.2 the Wan VAE's tiled encode/decode ignores
`config.patch_size`, which the Wan 2.2 VAE needs, so `vae_tiling=true`
crashed; without tiling the decode runs out of memory on the L4. 0.36.0 fixes
the tiling. Only the diffusers pin changed: pip's resolver accepts 0.36.0
with the same transformers, accelerate and huggingface_hub versions (checked
2026-10-04 for Linux / Python 3.11). torch 2.8.0+cu126 is confirmed working
on the L4 node (driver 580.178.04). diffusers 0.36.0 itself has not been run
on Camber yet.

## Step 4 (Camber): switch to the GPU node

Stop the Xxsmall node and start the **GPU · Xsmall** node. The budget clock
is running from now on.

## Step 5 (Camber, GPU node): smoke run first, then the full run

**Always start a GPU session with a smoke run.** It goes through every stage
(load, 2 denoising steps, VAE decode, encoding, measurement) in about 8–10
minutes. If something is broken you find out then, not after 35 minutes.

```python
import runpy
code = runpy.run_path("/home/jovyan/ai-animation-studio/scripts/camber_run.py")["main"](["--seed", "42", "--smoke"])
print("exit code:", code)
```

If it ends with `status: success` and `exit code: 0`, run the full benchmark
**in the same session**:

```python
import runpy
code = runpy.run_path("/home/jovyan/ai-animation-studio/scripts/camber_run.py")["main"](["--seed", "42"])
print("exit code:", code)
```

(These are the same as `python3 scripts/camber_run.py --seed 42 [--smoke]`.)

A smoke run is recorded like any run, with
`num_inference_steps=2` and the note `SMOKE TEST - not for evaluation`. Do
not rate it.

**Defaults** (chosen for the L4, each can be overridden):

| Setting | Default | Why | Override |
|---|---|---|---|
| benchmark / backend / size | `comic_rooftop_01` / `wan-ti2v-5b` / 1280×704 | | `--benchmark`, `--backend`, `--width`, `--height` |
| `cpu_offload` | `true` | transformer, text encoder and VAE do not fit in the L4's memory together | `--param cpu_offload=false` |
| `vae_tiling` | `true` | without it the VAE decode ran out of memory (2.60 GiB requested with 18.34 GiB allocated) | `--param vae_tiling=false` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | less memory fragmentation | set the variable yourself before the cell (`os.environ[...] = ...`); your value is kept |

What a run cell does:

1. prints `nvidia-smi`. If no GPU is visible it stops with exit code 2 and
   tells you to start the GPU Xsmall node; nothing is run
2. prints the effective settings: mode (smoke or full), backend parameters,
   `PYTORCH_CUDA_ALLOC_CONF`, `HF_HOME`, `HF_HUB_OFFLINE`
3. runs the benchmark with the environment's python, with
   `AI_STUDIO_ROOT` = the project, `HF_HOME=/home/jovyan/hf` and
   `HF_HUB_OFFLINE=1` (the weights are read from disk, never downloaded)
4. zips the record and the run's output folder and prints the zip path

`PYTORCH_CUDA_ALLOC_CONF`, `CUDA_VISIBLE_DEVICES` and `HF_HUB_OFFLINE` are
stored in every record (`environment.env_vars`), for failed runs too, so the
allocator setting of a run can always be looked up.

**Checking progress while the cell is silent.** The cell prints nothing
until it finishes. In the file browser open
`ai-animation-studio/outputs/comic_rooftop_01/` and refresh:

- a new `<run_id>/` folder appears when the model has loaded (after ~6 min)
- `start_frame_1280x704.png` in it means denoising has started
- `out.mp4` appears at the very end, after the decode

Exit codes: `0` success; `1` the run failed and **was recorded** as a failed
run (the stage and error are printed, and there is still a zip to download);
`2` setup problem, nothing recorded.

More options, as extra list items:

```python
["--seed", "42", "--param", "num_inference_steps=30", "--notes", "30-step comparison"]
```

## Step 6: STOP the GPU node

As soon as the last run cell has finished, **stop the GPU · Xsmall node**.
Do this before downloading anything: the zips are on the persistent disk and
stay there, and the budget is spent for as long as the node is on.

## Step 7 (Camber, CPU node "Xxsmall" or none): download the run zips

Stash files persist and downloading needs no GPU. In the file browser go to
`ai-animation-studio/dist/` and download `run-comic_rooftop_01-<run_id>.zip`
for each run (the smoke run too, if you want its record). Each zip contains
the record JSON and the output folder (`out.mp4`, the derived start frame).

To zip an earlier run again (the newest run of the benchmark by default; add
`"--run-id", "<run_id>"` for a specific one), on the CPU node:

```python
import runpy
runpy.run_path("/home/jovyan/ai-animation-studio/scripts/collect_run.py")["main"](["--benchmark", "comic_rooftop_01"])
```

## Step 8 (Windows): import the run

```
python scripts\import_run.py %USERPROFILE%\Downloads\run-comic_rooftop_01-<run_id>.zip
```

The files are created at the same project-relative paths they had on Camber
(`benchmarks/results/comic_rooftop_01/<run_id>.json`,
`outputs/comic_rooftop_01/<run_id>/...`) and listed. Nothing is ever
overwritten: if any of the files already exists, nothing is written. Zip
entries outside `benchmarks/results/` and `outputs/` are rejected.

Then commit the new record JSON (results are tracked in git; videos are
not), and rate the full run as described in
[BENCHMARK_COMIC_ROOFTOP_01.md](BENCHMARK_COMIC_ROOFTOP_01.md). Failed runs
are benchmark data too: import and commit their records as well.

## Updating the code on Camber

Whenever the project changed on Windows:

1. commit, then `python scripts\make_bundle.py`
2. on the **Xxsmall** CPU node: upload the bundle, run the **cleanup** cell,
   then the **unzip and verify** cell (step 2)
3. run `camber_setup.py` again (step 3) if `requirements/camber-wan.txt`
   changed. It did for this version: the diffusers pin went from 0.35.2 to
   0.36.0. When in doubt, run it; it is cheap on the CPU node
4. only then switch to the GPU node

## What happened in the first session

Two runs on the L4 (driver 580.178.04, CUDA 13.0, 22.03 GiB usable), both
with `cpu_offload=true` and 4 steps:

1. No tiling: the weights loaded in 6 min 18 s, the 4 denoising steps ran at
   ~30 s per step, then the VAE decode failed with CUDA out of memory ("Tried
   to allocate 2.60 GiB", 18.34 GiB allocated by PyTorch).
2. With `vae_tiling=true` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
   failed at the start of generation with "weight of size [160, 12, 3, 3, 3],
   expected input[1, 3, 3, 258, 258] to have 12 channels, but got 3 channels".
   The tiled VAE code in diffusers 0.35.2 ignores the Wan 2.2 VAE's
   `patch_size`.

Fixes in this version: diffusers 0.36.0; `vae_tiling=true` and the allocator
setting are now defaults of `camber_run.py`; the adapter refuses
`vae_tiling=true` on diffusers older than 0.36.0 with a clear message; the
allocator setting is stored in every record.

## Known limits

- **Only a 2-step smoke run has succeeded so far**; see the status note at
  the top.
- **`git_dirty` and earlier runs.** Record, metrics and output files of
  earlier runs no longer make a run "dirty": `git_dirty` is true only when
  something outside the data folders changed. `git_untracked_data_only` and
  the raw `git_status` lines are recorded as well. Records made before this
  change may show `git_dirty: true` just because older records were present.
- With `vae_tiling=true` the decoded pixels can differ slightly from an
  untiled decode. All L4 runs use it, so they are comparable with each
  other; the setting is in every record.
- The Camber-specific paths (`/home/jovyan/...`) are constants at the top of
  `scripts/camber_setup.py` and `scripts/camber_run.py`.

## Running without Camber

The launcher itself is not Camber-specific. On any machine with the `wan`
extra installed, a CUDA GPU, ffmpeg and ffprobe:

```
python scripts/run_benchmark.py --benchmark comic_rooftop_01 --backend wan-ti2v-5b --seed 42 --width 1280 --height 704 --param cpu_offload=true --param vae_tiling=true
```

`--param key=value` values are parsed as JSON when possible (`true`, `50`,
`4.5`), otherwise kept as text. See `src/ai_studio/cli.py`.
