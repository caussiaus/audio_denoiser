#!/usr/bin/env python3
"""
stitch.py
– Load 44.1 kHz MP3 segments with torchaudio.load
– Resample/convert to mono
– Concatenate tensors in chronological order
– Save mapping of segment lengths for de-stitching
"""
# TODO: implement using torchaudio and torch.cat
