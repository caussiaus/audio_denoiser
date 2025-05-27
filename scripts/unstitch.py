#!/usr/bin/env python3
"""
unstitch.py
– Load enhanced waveform tensor
– Read original segment-length mapping
– Slice enhanced tensor back into individual segments
– Save each segment as its own 44.1 kHz MP3/WAV
"""
#!/usr/bin/env python3
import torch
import torchaudio
import json
import sys
from pathlib import Path

def main(enhanced_wav, mapping_json, out_dir):
    """
    enhanced_wav: denoised stitched WAV
    mapping_json: mapping.json from stitch.py
    out_dir: directory to write per-segment files
    """
    os.makedirs(out_dir, exist_ok=True)
    enhanced, sr = torchaudio.load(enhanced_wav)
    mapping = json.load(open(mapping_json))
    lengths, names = mapping["lengths"], mapping["filenames"]

    cursor = 0
    for length, name in zip(lengths, names):
        segment = enhanced[:, cursor:cursor + length]
        cursor += length
        base, _ = Path(name).stem, Path(name).suffix
        out_path = Path(out_dir) / f"{base}_denoised{base}.wav"
        torchaudio.save(str(out_path), segment, sr)
        print(f"Saved segment → {out_path}")

    assert cursor == enhanced.shape[1], "Length mismatch: did not consume all samples"

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: unstitch.py enhanced.wav mapping.json output_folder")
        sys.exit(1)
    main(*sys.argv[1:])
