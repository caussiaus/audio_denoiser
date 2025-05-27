#!/usr/bin/env python3
"""
stitch.py
– Load 44.1 kHz MP3 segments with torchaudio.load
– Resample/convert to mono
– Concatenate tensors in chronological order
– Save mapping of segment lengths for de-stitching
"""
#!/usr/bin/env python3
import torch
import torchaudio
import json
import sys
from pathlib import Path

def load_and_preprocess(file_path, target_sr=44100):
    """Load audio, convert to mono, resample to target_sr."""
    waveform, sr = torchaudio.load(file_path)
    # Stereo → mono by averaging
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr

def main(manifest_json, out_stitched, out_map, target_sr=44100):
    """
    manifest_json: JSON list of input file paths in chronological order
    out_stitched: path to save concatenated WAV
    out_map: path to save JSON mapping of segment lengths and original filenames
    """
    manifest = json.load(open(manifest_json))
    segments = []
    lengths = []
    filenames = []

    for fp in manifest:
        wav, sr = load_and_preprocess(fp, target_sr)
        segments.append(wav)
        lengths.append(wav.shape[1])
        filenames.append(Path(fp).name)

    # Concatenate along time axis
    stitched = torch.cat(segments, dim=1)
    # Save stitched file (optional WAV preview)
    torchaudio.save(out_stitched, stitched, target_sr)
    # Save mapping
    mapping = {"lengths": lengths, "filenames": filenames, "sample_rate": target_sr}
    with open(out_map, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Stitched {len(segments)} segments → {out_stitched}")
    print(f"Mapping saved → {out_map}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: stitch.py manifest.json out_stitched.wav mapping.json")
        sys.exit(1)
    main(*sys.argv[1:])
