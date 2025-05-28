#!/usr/bin/env python3
"""
audio_pipeline.py — Full MP3-only pipeline: stitch, enhance with DNS64, unstitch → MP3 outputs.

Usage:
  python scripts/audio_pipeline.py \
    --input_dir audios \
    --output_dir output \
    --sr 16000 \
    --chunk_secs 30 \
    --overlap_secs 1
"""

import os
import torch
import torchaudio
from denoiser import pretrained
from pathlib import Path
import argparse
from tqdm import tqdm

# optional: to read ID3 tags
try:
    from mutagen.easyid3 import EasyID3
except ImportError:
    EasyID3 = None

def gather_sources(src_dir, exts=(".mp3", ".wav", ".flac", ".m4a")):
    p = Path(src_dir)
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])

def load_and_preprocess(file_path: Path, target_sr: int):
    wav, sr = torchaudio.load(str(file_path))
    # downmix to mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # resample if needed
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def process_chunk(chunk: torch.Tensor, model, device):
    # chunk: [1, T]
    chunk = chunk.unsqueeze(0).to(device)      # → [1,1,T]
    with torch.no_grad():
        enhanced = model(chunk)                # → [1,1,T]
    return enhanced.squeeze(0).cpu()           # → [1,T]

def extract_title(fp: Path):
    if EasyID3:
        try:
            tags = EasyID3(str(fp))
            return tags.get("title", [fp.stem])[0]
        except Exception:
            return fp.stem
    else:
        return fp.stem

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True,  help="Folder of audio files to process")
    p.add_argument("--output_dir", default="enhanced_outputs", help="Where to save denoised files")
    p.add_argument("--sr",         type=int, default=16000, help="Model sample‐rate (Hz)")
    p.add_argument("--chunk_secs", type=int, default=30,   help="Chunk length in seconds")
    p.add_argument("--overlap_secs",type=int, default=1,    help="Seconds overlap between chunks")
    args = p.parse_args()

    files = gather_sources(args.input_dir)
    if not files:
        print("❌ No audio files found in input_dir.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pretrained.dns64().to(device).eval()

    # 1) Load & stitch everything into one long [1, total_samples] tensor
    segments, lengths, stems = [], [], []
    for f in files:
        wav = load_and_preprocess(f, args.sr)
        segments.append(wav)
        lengths.append(wav.size(1))
        stems.append(extract_title(f))
    stitched = torch.cat(segments, dim=1)

    # 2) Enhance in overlapping chunks
    chunk_size = args.chunk_secs * args.sr
    overlap    = args.overlap_secs * args.sr
    total_len  = stitched.size(1)

    enhanced_chunks = []
    for start in tqdm(range(0, total_len, chunk_size - overlap), desc="Enhancing"):
        end = min(start + chunk_size, total_len)
        chunk = stitched[:, start:end]
        if chunk.size(1) < chunk_size:
            pad = torch.zeros(1, chunk_size - chunk.size(1))
            chunk = torch.cat([chunk, pad], dim=1)
        out_chunk = process_chunk(chunk, model, device)
        # trim padding back to original (end-start) length
        out_chunk = out_chunk[:, : end - start]
        enhanced_chunks.append(out_chunk)

    # 3) Recombine, **dropping** the overlap regions on all but the first chunk
    enhanced = enhanced_chunks[0]
    for chunk in enhanced_chunks[1:]:
        enhanced = torch.cat([enhanced, chunk[:, overlap:]], dim=1)

    # Sanity check
    assert enhanced.size(1) == sum(lengths), \
        f"Length mismatch: got {enhanced.size(1)} vs {sum(lengths)}"

    # 4) Unstitch & save each segment as MP3 with "<title>_denoised.mp3"
    cursor = 0
    for length, title in zip(lengths, stems):
        seg = enhanced[:, cursor: cursor + length]
        cursor += length
        out_fn = f"{title}_denoised.mp3"
        out_path = Path(args.output_dir) / out_fn
        torchaudio.save(str(out_path), seg, args.sr, format="mp3")
        print(f"✔️ Saved {out_fn}")

    print(f"\n✅ Done! All denoised files in: {args.output_dir}")

if __name__ == "__main__":
    main()
