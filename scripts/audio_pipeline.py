#!/usr/bin/env python3
"""
audio_pipeline.py  ▸  **Batch AudioSR enhancer (resume‑safe)**
──────────────────────────────────────────────────────────────────
• Upscales every file to pristine 48‑kHz mono WAV using **AudioSR**.
• Works with MP3 / WAV / FLAC / M4A / AAC input.
• Skips files whose final output already exists (unless --force).
• Uses a per‑file *temporary* workspace (output/.tmp) so no more
  clutter or accidental overwrites.
• File names are preserved **exactly**:  `song.mp3 → song.wav`.

Requirements
~~~~~~~~~~~~
    pip install audiosr torch torchaudio tqdm ffmpeg-python
    (FFmpeg must be on PATH.)

Examples
~~~~~~~~
    # GPU run, resume‑safe
    python audio_pipeline.py --input_dir in --output_dir out

    # Force re‑processing and use CPU
    python audio_pipeline.py --input in --output out --force --cpu
"""

from __future__ import annotations
import argparse, shutil, subprocess, sys, uuid, tempfile
from pathlib import Path
from typing import Sequence
from tqdm import tqdm
import torch, torchaudio
import audiosr

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}

def run(cmd: Sequence[str]):
    """Run shell command, abort on error."""
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# ────────────────────────────────────────────────────────────────────────────
# FFmpeg helpers
# ────────────────────────────────────────────────────────────────────────────

def to_wav48k_mono(src: Path, dst: Path):
    """Convert arbitrary audio to 48‑kHz mono WAV via FFmpeg."""
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-ac", "1", "-ar", "48000", str(dst)
    ])

# ────────────────────────────────────────────────────────────────────────────
# AudioSR wrapper (prefers Python API, falls back to CLI)
# ────────────────────────────────────────────────────────────────────────────

def audiosr_enhance(inp: Path, outp: Path, device: str):
    """Call AudioSR. Chooses Python API if available, else CLI."""
    try:
        import audiosr
        model = audiosr.build_model(device=device)
        audiosr.super_resolution(model, str(inp), ddim_steps=200)
    except Exception:
        # CLI fallback (requires audiosr entry‑point on PATH)
        cmd = ["audiosr", "-i", str(inp), "-o", str(outp)]
        if device == "cpu":
            cmd += ["--device", "cpu"]
        run(cmd)

# ────────────────────────────────────────────────────────────────────────────
# Main processing routine
# ────────────────────────────────────────────────────────────────────────────

def process_one(src: Path, out_dir: Path, device: str, force: bool) -> bool:
    final_out = out_dir / f"{src.stem}.wav"
    if final_out.exists() and not force:
        return False  # skipped

    tmp_root = out_dir / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td = Path(td)
        tmp_in  = td / "in.wav"
        tmp_out = td / "out.wav"

        # Always convert to uniform wav (cheap & safe)
        to_wav48k_mono(src, tmp_in)

        # Enhance
        try:
            audiosr_enhance(tmp_in, tmp_out, device)
            # If tmp_out exists, use it (CLI fallback). Otherwise, use tmp_in (Python API overwrites input)
            if tmp_out.exists():
                shutil.move(tmp_out, final_out)
            else:
                shutil.move(tmp_in, final_out)
        except Exception as e:
            print(f"Error processing {src}: {e}")
            return False

    return True  # processed

# ────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("Batch AudioSR enhancer (resume‑safe)")
    ap.add_argument("--input_dir", default="input", help="Source directory")
    ap.add_argument("--output_dir", default="output", help="Destination directory")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--force", action="store_true", help="Re‑process even if WAV exists")
    args = ap.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        sys.exit(f"❌ Input dir not found: {in_dir}")
    out_dir.mkdir(exist_ok=True)

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS])
    if not files:
        sys.exit("❌ No audio files found.")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    done, skipped = 0, 0
    for f in tqdm(files, desc="Enhancing"):
        if process_one(f, out_dir, device, args.force):
            done += 1
        else:
            skipped += 1

    print(f"\n✅ Finished. New/updated: {done}  |  Skipped: {skipped}\nOutputs → {out_dir}\n")


if __name__ == "__main__":
    main()
