# Running a benchmark on Camber (from Windows)

Camber gives a free single NVIDIA L4 (24 GB) GPU node. This guide takes the
project from your Windows machine to Camber, runs `comic_rooftop_01` on
Wan 2.2 TI2V-5B there, and brings the result back.

> **Status: this workflow has not been run end to end yet.** The scripts are
> tested locally with fakes (no conda, no GPU). The first real run will be the
> first time the conda setup, the pinned packages and the model itself are
> exercised on Camber. If a step fails, the output of that cell is what is
> needed to fix it.

## GPU budget: read this first

- The free GPU budget is **5 GPU-hours per month**.
- GPU time is counted **while the GPU node is on**, not while a run is
  executing. An idle GPU node still spends the budget.
- So: do everything you can on the **CPU node** (upload, unzip, setup), start
  the **GPU Xsmall** node only for the run itself, and **stop it** as soon as
  you have downloaded the result.

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

All notebook cells below are plain Python. Do not use `%%bash` cells: text
pasted from Windows carries CRLF line endings, which breaks them.

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

## Step 2 (Camber, CPU node): upload and unzip

Start a **CPU** node. Upload `ai-animation-studio-bundle.zip` with the
**Stash upload button** so that it lands in `/home/jovyan`. If the upload
puts it somewhere else, change `BUNDLE` in the cell.

```python
import zipfile
BUNDLE = "/home/jovyan/ai-animation-studio-bundle.zip"
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall("/home/jovyan")
print("unpacked to /home/jovyan/ai-animation-studio")
```

Updating later with a newer bundle: run the same cell. Files are overwritten
in place and `benchmarks/results/`, `outputs/` and `dist/` on Camber are left
alone. Files that were deleted from the project since the last bundle stay
behind; to remove them, run this before unzipping (it keeps results and
outputs):

```python
import shutil
for name in ("src", "scripts", "tests", "docs", "requirements", ".git"):
    shutil.rmtree(f"/home/jovyan/ai-animation-studio/{name}", ignore_errors=True)
```

## Step 3 (Camber, CPU node): set up the environment

Needed once, and again only when `requirements/camber-wan.txt` changes.
It is safe to run again: finished steps are skipped or do nothing.

```python
import runpy
code = runpy.run_path("/home/jovyan/ai-animation-studio/scripts/camber_setup.py")["main"]()
print("exit code:", code)
```

(This is the same as `python3 scripts/camber_setup.py` with the base python.)

What it does:

1. creates the conda env `/home/jovyan/envs/ai-studio` with Python 3.11
   (from conda-forge), with conda's package cache in `/tmp/conda-pkgs` and
   pip's cache in `/tmp/pip-cache` so no cache lands on the 50 GB quota
2. `pip install -r requirements/camber-wan.txt`
3. `pip install -e . --no-deps` (the project itself)
4. verifies `import torch, torchvision, diffusers, transformers, ai_studio`
   and prints the versions
5. prints the size of `/home/jovyan/envs`, `/home/jovyan/hf` and the whole
   home directory, and warns above 45 GB

It must end with `SETUP OK` and `exit code: 0`. `cuda available False` in the
version line is expected on the CPU node. On any error it stops and prints
`SETUP FAILED: ...`.

Pinned packages (`requirements/camber-wan.txt`, Python 3.11, Linux):

| Package | Version |
|---|---|
| torch | 2.8.0+cu126 (PyTorch index `https://download.pytorch.org/whl/cu126`) |
| torchvision | 0.23.0+cu126 |
| diffusers | 0.35.2 |
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

These versions exist for Linux / Python 3.11 and resolve together without
conflicts (checked with pip's resolver on 2026-10-04). They have **not** been
installed or run on Camber yet. The NVIDIA driver version on the L4 node is
not known here: if the run cell reports that CUDA is not available on the GPU
node, the driver is too old for the CUDA 12.6 build. The header of
`requirements/camber-wan.txt` gives the CUDA 12.4 alternative
(`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`).

## Step 4 (Camber): switch to the GPU node

Stop the CPU node and start the **GPU Xsmall** node (L4, 8 CPU, 32 GB RAM).
The budget clock is running from now on.

## Step 5 (Camber, GPU node): run the benchmark

```python
import runpy
code = runpy.run_path("/home/jovyan/ai-animation-studio/scripts/camber_run.py")["main"](["--seed", "42"])
print("exit code:", code)
```

(This is the same as `python3 scripts/camber_run.py --seed 42`.)

Defaults: benchmark `comic_rooftop_01`, backend `wan-ti2v-5b`, 1280×704,
`cpu_offload=true`. The transformer (~10 GB in bf16), the text encoder
(~11.4 GB) and the VAE (~2.8 GB) do not fit in 24 GB together, so on the L4
the components are moved to the GPU one at a time.

What it does:

1. prints `nvidia-smi`. If no GPU is visible it stops with exit code 2 and
   tells you to start the GPU Xsmall node; nothing is run
2. runs the benchmark with the environment's python, with
   `AI_STUDIO_ROOT` = the project, `HF_HOME=/home/jovyan/hf` and
   `HF_HUB_OFFLINE=1` (the weights are read from disk, never downloaded),
   showing the output live
3. zips the record and the run's output folder and prints the zip path

Exit codes: `0` success; `1` the run failed and **was recorded** as a failed
run (the stage and error are printed, and there is still a zip to download);
`2` setup problem, nothing recorded.

More options, as extra list items:

```python
["--seed", "42", "--param", "num_inference_steps=30", "--notes", "first L4 run"]
```

If the run fails with an out-of-memory error while decoding the video, retry
with VAE tiling; no code change is needed:

```python
["--seed", "42", "--param", "vae_tiling=true"]
```

How long a run takes on the L4 is not known yet. The first run will tell;
its record has the timing breakdown (model load vs. inference).

## Step 6 (Camber): download the run zip

In the file browser go to `ai-animation-studio/dist/`, and download
`run-comic_rooftop_01-<run_id>.zip` (right-click → Download). It contains the
record JSON and the output folder (`out.mp4`, the derived start frame).

To zip an earlier run again (the newest run of the benchmark by default):

```python
import runpy
runpy.run_path("/home/jovyan/ai-animation-studio/scripts/collect_run.py")["main"](["--benchmark", "comic_rooftop_01"])
```

## Step 7: STOP the GPU node

Do it now, before looking at the video. The budget is spent while the node
is on.

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
not), and rate the run as described in
[BENCHMARK_COMIC_ROOFTOP_01.md](BENCHMARK_COMIC_ROOFTOP_01.md).

## Known limits

- **`git_dirty` after the first run.** Record files are untracked until you
  commit them on Windows, so from the second run on Camber, records show
  `git_dirty: true` even though the code is unchanged. Upload a fresh bundle
  (made after committing the imported results) to get a clean tree again.
- Nothing here has been run on Camber yet; see the status note at the top.
- The Camber-specific paths (`/home/jovyan/...`) are constants at the top of
  `scripts/camber_setup.py` and `scripts/camber_run.py`.

## Running without Camber

The launcher itself is not Camber-specific. On any machine with the `wan`
extra installed, a CUDA GPU, ffmpeg and ffprobe:

```
python scripts/run_benchmark.py --benchmark comic_rooftop_01 --backend wan-ti2v-5b --seed 42 --width 1280 --height 704 --param cpu_offload=true
```

`--param key=value` values are parsed as JSON when possible (`true`, `50`,
`4.5`), otherwise kept as text. See `src/ai_studio/cli.py`.
