#!/usr/bin/env python3
"""
enhance.py
– Load stitched waveform tensor
– Move to CUDA
– Run through a pretrained denoising model (e.g. denoiser.dns64())
– Move result back to CPU
– Save enhanced waveform for splitting
"""
#!/usr/bin/env python3
import torch
import torchaudio
import json
import sys
from denoiser import pretrained

def main(stitched_wav, mapping_json, out_enhanced):
    """
    stitched_wav: path to stitched WAV
    mapping_json: same JSON from stitch.py
    out_enhanced: path to save denoised WAV
    """
    # Load stitched
    waveform, sr = torchaudio.load(stitched_wav)
    # Move to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    waveform = waveform.to(device)
    # Load DNS64 model (48 kHz) or swap e.g. dns48()
    model = pretrained.dns64().to(device).eval()
    # Inference
    with torch.no_grad():
        enhanced = model(waveform)
    # Some models add latency—here, we assume none or ignore small delay
    enhanced = enhanced.cpu()
    # Save enhanced full waveform
    torchaudio.save(out_enhanced, enhanced, sr)
    print(f"Enhanced audio saved → {out_enhanced}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: enhance.py stitched.wav mapping.json enhanced.wav")
        sys.exit(1)
    main(*sys.argv[1:])
