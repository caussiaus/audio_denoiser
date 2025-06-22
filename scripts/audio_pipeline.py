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
    
    # Use VAD-based denoising for better noise reduction
    python audio_pipeline.py --input_dir in --output_dir out --use_vad_denoising
"""

from __future__ import annotations
import argparse, shutil, subprocess, sys, uuid, tempfile
from pathlib import Path
from typing import Sequence, List, Tuple, Optional
from tqdm import tqdm
import torch, torchaudio
import numpy as np

# Metadata handling
try:
    from mutagen import File as MetaFile
    from mutagen.easyid3 import EasyID3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("Warning: mutagen not available. Metadata preservation disabled.")

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}

def get_metadata_title(src: Path) -> str:
    """Extract title from audio file metadata, fallback to filename stem."""
    if not MUTAGEN_AVAILABLE:
        return src.stem
    
    try:
        mf = MetaFile(str(src))
        if mf is None:
            return src.stem
            
        # Try different tag formats
        if hasattr(mf, 'tags') and mf.tags:
            tags = mf.tags
            # Try common title fields
            for key in ['title', 'TIT2', 'TITLE']:
                if key in tags:
                    value = tags[key]
                    if isinstance(value, list) and value:
                        return str(value[0]).strip()
                    elif value:
                        return str(value).strip()
        
        # Try EasyID3 for MP3 files
        if src.suffix.lower() == '.mp3':
            try:
                easy_tags = EasyID3(str(src))
                if 'title' in easy_tags:
                    return str(easy_tags['title'][0]).strip()
            except:
                pass
                
    except Exception as e:
        print(f"Warning: Could not read metadata from {src}: {e}")
    
    return src.stem

def encode_mp3_with_metadata(wav_path: Path, mp3_path: Path, title: str, lufs: int = -23):
    """Encode WAV to MP3 with metadata and loudness normalization."""
    # Create safe filename from title
    safe_title = ''.join(c for c in title if c.isalnum() or c in ' _-').strip()
    safe_title = safe_title.replace(' ', '_')
    
    # Ensure the output path uses the safe title
    mp3_dir = mp3_path.parent
    mp3_path = mp3_dir / f"{safe_title}.mp3"
    
    # Encode with loudness normalization and metadata
    cmd = [
        "/usr/bin/ffmpeg", "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-af", f"loudnorm=I={lufs}:LRA=7:TP=-2",
        "-c:a", "libmp3lame", "-b:a", "320k",
        "-metadata", f"title={title}",
        "-id3v2_version", "3",
        str(mp3_path)
    ]
    
    try:
        run(cmd)
        return mp3_path
    except Exception as e:
        print(f"Error encoding MP3: {e}")
        return None

def run(cmd: Sequence[str]):
    """Run shell command, abort on error."""
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# ────────────────────────────────────────────────────────────────────────────
# FFmpeg helpers
# ────────────────────────────────────────────────────────────────────────────

def to_wav48k_mono(src: Path, dst: Path):
    """Convert arbitrary audio to 48‑kHz mono WAV via FFmpeg."""
    run([
        "/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-ac", "1", "-ar", "48000", str(dst)
    ])

# ────────────────────────────────────────────────────────────────────────────
# VAD-based denoising functions
# ────────────────────────────────────────────────────────────────────────────

def load_vad_model(device: str):
    """Load Silero VAD model."""
    try:
        import torch
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=False,
                                    onnx=False)
        model.to(device)
        return model, utils
    except Exception as e:
        print(f"Warning: Could not load VAD model: {e}")
        return None, None

def detect_speech_segments(audio_path: Path, device: str, min_speech_duration_ms: int = 250, 
                          min_silence_duration_ms: int = 100, speech_pad_ms: int = 30):
    """Detect speech segments using Silero VAD, robust to return signature changes. Always uses CPU for VAD."""
    # Always use CPU for VAD
    model, utils = load_vad_model('cpu')
    if model is None:
        return None
    
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        vad_sample_rate = 16000
        if sample_rate != vad_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, vad_sample_rate)
            vad_waveform = resampler(waveform)
        else:
            vad_waveform = waveform
        vad_waveform = vad_waveform.cpu()  # ensure on CPU

        if isinstance(utils, dict):
            get_speech_timestamps = utils.get('get_speech_timestamps')
        elif isinstance(utils, (list, tuple)):
            get_speech_timestamps = next((u for u in utils if callable(u)), None)
        else:
            get_speech_timestamps = utils

        if get_speech_timestamps is None or not callable(get_speech_timestamps):
            raise RuntimeError("Could not locate get_speech_timestamps in Silero utils")

        try:
            speech_timestamps = get_speech_timestamps(
                vad_waveform, model,
                sampling_rate=vad_sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms)
        except TypeError:
            speech_timestamps = get_speech_timestamps(vad_waveform, model, sampling_rate=vad_sample_rate)

        print(f"[VAD] speech_timestamps type: {type(speech_timestamps)}, len: {len(speech_timestamps) if hasattr(speech_timestamps, '__len__') else 'N/A'}")
        if isinstance(speech_timestamps, tuple):
            print(f"[VAD] speech_timestamps is tuple, using first element.")
            speech_timestamps = speech_timestamps[0]

        # Convert [{'start': x, 'end': y}, ...] to [(x, y), ...]
        if speech_timestamps and isinstance(speech_timestamps[0], dict):
            speech_timestamps = [(d['start'], d['end']) for d in speech_timestamps]

        if sample_rate != vad_sample_rate:
            scale_factor = sample_rate / vad_sample_rate
            speech_timestamps = [(int(start * scale_factor), int(end * scale_factor)) 
                               for start, end in speech_timestamps]

        return speech_timestamps, waveform, sample_rate
    except Exception as e:
        print(f"Warning: VAD detection failed: {e}")
        return None

def apply_dns64_denoising(audio_path: Path, device: str, num_passes: int = 1):
    """Apply DNS64 denoising to audio, possibly multiple times."""
    try:
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
        
        # Load DNS64 model
        model = pretrained.dns64().to(device)
        
        # Load and convert audio
        wav, sr = torchaudio.load(str(audio_path))
        wav = convert_audio(wav, sr, model.sample_rate, model.chin)
        
        # Apply denoising multiple times if needed
        with torch.no_grad():
            for _ in range(num_passes):
                wav = model(wav.unsqueeze(0).to(device))[0].cpu()
        
        return wav
    except Exception as e:
        print(f"Warning: DNS64 denoising failed: {e}")
        return None

def crossfade_audio(audio1: torch.Tensor, audio2: torch.Tensor, crossfade_length: int) -> torch.Tensor:
    """Apply equal-power crossfade between two audio segments."""
    if crossfade_length <= 0:
        return audio1
    
    # Create crossfade windows
    fade_out = torch.sqrt(torch.linspace(1.0, 0.0, crossfade_length))
    fade_in = torch.sqrt(torch.linspace(0.0, 1.0, crossfade_length))
    
    # Apply crossfade
    result = torch.zeros_like(audio1)
    result[:crossfade_length] = audio1[:crossfade_length] * fade_out + audio2[:crossfade_length] * fade_in
    
    return result

def vad_denoise_audio(audio_path: Path, device: str, denoise_strength_silence: int = 3, denoise_strength_speech: int = 1, target_noise_floor: float = 0.0005):
    """Apply VAD-based adaptive denoising to audio."""
    vad_result = detect_speech_segments(audio_path, device)
    if vad_result is None:
        print("VAD detection failed, using original audio")
        return None
    
    speech_timestamps, waveform, sample_rate = vad_result
    if not speech_timestamps:
        print("No speech detected, applying full denoising")
        denoised = apply_dns64_denoising(audio_path, device, num_passes=denoise_strength_silence)
        if denoised is not None:
            return denoised
        return waveform.squeeze()
    
    processed_segments = []
    crossfade_samples = int(0.1 * sample_rate)  # 100ms crossfade
    # Helper to measure noise floor
    def noise_floor(segment):
        arr = segment.cpu().numpy().flatten()
        return np.percentile(np.abs(arr), 95)
    # Initial silence
    if speech_timestamps[0][0] > 0:
        initial_silence_end = speech_timestamps[0][0]
        if initial_silence_end > crossfade_samples:
            initial_silence = waveform[:, :initial_silence_end]
            temp_silence_path = audio_path.parent / "temp_initial_silence.wav"
            torchaudio.save(str(temp_silence_path), initial_silence, sample_rate)
            # Adaptive denoising
            denoised = initial_silence
            for _ in range(denoise_strength_silence):
                denoised = apply_dns64_denoising(temp_silence_path, device, num_passes=1)
                torchaudio.save(str(temp_silence_path), denoised, sample_rate)
                if noise_floor(denoised) <= target_noise_floor:
                    break
            processed_segments.append((0, denoised.squeeze()))
            temp_silence_path.unlink(missing_ok=True)
    # Speech segments
    for i, (start, end) in enumerate(speech_timestamps):
        pad_samples = int(0.1 * sample_rate)
        segment_start = max(0, start - pad_samples)
        segment_end = min(len(waveform[0]), end + pad_samples)
        segment = waveform[:, segment_start:segment_end]
        temp_segment_path = audio_path.parent / f"temp_segment_{i}.wav"
        torchaudio.save(str(temp_segment_path), segment, sample_rate)
        # Gentle denoising for speech
        denoised = apply_dns64_denoising(temp_segment_path, device, num_passes=denoise_strength_speech)
        if denoised is not None:
            processed_segments.append((segment_start, denoised.squeeze()))
        else:
            processed_segments.append((segment_start, segment.squeeze()))
        temp_segment_path.unlink(missing_ok=True)
    # Final silence
    last_speech_end = speech_timestamps[-1][1]
    if last_speech_end < len(waveform[0]):
        final_silence_start = last_speech_end
        final_silence = waveform[:, final_silence_start:]
        if len(final_silence[0]) > crossfade_samples:
            temp_silence_path = audio_path.parent / "temp_final_silence.wav"
            torchaudio.save(str(temp_silence_path), final_silence, sample_rate)
            denoised = final_silence
            for _ in range(denoise_strength_silence):
                denoised = apply_dns64_denoising(temp_silence_path, device, num_passes=1)
                torchaudio.save(str(temp_silence_path), denoised, sample_rate)
                if noise_floor(denoised) <= target_noise_floor:
                    break
            processed_segments.append((final_silence_start, denoised.squeeze()))
            temp_silence_path.unlink(missing_ok=True)
    # Reconstruct audio with crossfades
    if not processed_segments:
        return waveform.squeeze()
    processed_segments.sort(key=lambda x: x[0])
    result = torch.zeros(len(waveform[0]))
    for i, (start, segment) in enumerate(processed_segments):
        end = start + len(segment)
        if i == 0:
            result[:end] = segment
        else:
            prev_end = processed_segments[i-1][0] + len(processed_segments[i-1][1])
            if start > prev_end:
                result[prev_end:start] = 0
                result[start:end] = segment
            else:
                overlap = prev_end - start
                if overlap > 0:
                    crossfade_len = min(overlap, crossfade_samples)
                    result[start:start+crossfade_len] = crossfade_audio(
                        result[start:start+crossfade_len], 
                        segment[:crossfade_len], 
                        crossfade_len
                    )
                    result[start+crossfade_len:end] = segment[crossfade_len:]
                else:
                    result[start:end] = segment
    return result

# ────────────────────────────────────────────────────────────────────────────
# Advanced noise analysis and denoising functions
# ────────────────────────────────────────────────────────────────────────────

def analyze_noise_levels(audio: torch.Tensor, sample_rate: int, segment_name: str = "audio"):
    """Analyze noise levels in audio segment."""
    audio_np = audio.cpu().numpy().flatten()
    
    # Calculate various noise metrics
    rms = np.sqrt(np.mean(audio_np**2))
    peak = np.max(np.abs(audio_np))
    noise_floor_95 = np.percentile(np.abs(audio_np), 95)
    noise_floor_99 = np.percentile(np.abs(audio_np), 99)
    
    # Spectral analysis for noise characteristics
    if len(audio_np) > 1024:
        from scipy import signal
        freqs, psd = signal.welch(audio_np, sample_rate, nperseg=1024)
        # Find dominant frequency components
        peak_freq_idx = np.argmax(psd)
        peak_freq = freqs[peak_freq_idx]
        # Calculate spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    else:
        peak_freq = 0
        spectral_centroid = 0
    
    print(f"  📊 {segment_name} Noise Analysis:")
    print(f"     RMS: {rms:.6f}")
    print(f"     Peak: {peak:.6f}")
    print(f"     Noise Floor (95%): {noise_floor_95:.6f}")
    print(f"     Noise Floor (99%): {noise_floor_99:.6f}")
    print(f"     Peak Freq: {peak_freq:.1f} Hz")
    print(f"     Spectral Centroid: {spectral_centroid:.1f} Hz")
    
    return {
        'rms': rms,
        'peak': peak,
        'noise_floor_95': noise_floor_95,
        'noise_floor_99': noise_floor_99,
        'peak_freq': peak_freq,
        'spectral_centroid': spectral_centroid
    }

def apply_spectral_subtraction(audio: torch.Tensor, sample_rate: int, noise_reduction_factor: float = 2.0):
    """Apply spectral subtraction denoising."""
    try:
        import numpy as np
        from scipy import signal
        
        # Convert to numpy
        audio_np = audio.cpu().numpy().flatten()
        
        # Estimate noise spectrum from first 0.5 seconds
        noise_samples = int(0.5 * sample_rate)
        if len(audio_np) > noise_samples:
            noise_spectrum = np.abs(np.fft.fft(audio_np[:noise_samples]))
            noise_spectrum = np.mean(noise_spectrum.reshape(-1, 1024), axis=1)
        else:
            noise_spectrum = np.abs(np.fft.fft(audio_np))
        
        # Apply spectral subtraction
        fft_size = 2048
        hop_size = fft_size // 4
        
        # STFT
        freqs, times, stft = signal.stft(audio_np, sample_rate, nperseg=fft_size, noverlap=fft_size-hop_size)
        
        # Spectral subtraction
        noise_spectrum_resized = np.interp(freqs, np.linspace(0, sample_rate//2, len(noise_spectrum)), noise_spectrum)
        noise_spectrum_resized = noise_spectrum_resized.reshape(-1, 1)
        
        # Subtract noise spectrum
        stft_magnitude = np.abs(stft)
        stft_phase = np.angle(stft)
        
        # Spectral subtraction with spectral floor
        spectral_floor = 0.01
        stft_magnitude_denoised = np.maximum(
            stft_magnitude - noise_reduction_factor * noise_spectrum_resized,
            spectral_floor * stft_magnitude
        )
        
        # Reconstruct
        stft_denoised = stft_magnitude_denoised * np.exp(1j * stft_phase)
        denoised_audio, _ = signal.istft(stft_denoised, sample_rate, nperseg=fft_size, noverlap=fft_size-hop_size)
        
        # Ensure same length
        if len(denoised_audio) > len(audio_np):
            denoised_audio = denoised_audio[:len(audio_np)]
        elif len(denoised_audio) < len(audio_np):
            denoised_audio = np.pad(denoised_audio, (0, len(audio_np) - len(denoised_audio)))
        
        return torch.tensor(denoised_audio, dtype=audio.dtype)
    except Exception as e:
        print(f"Warning: Spectral subtraction failed: {e}")
        return audio

def apply_rnnoise_denoising(audio: torch.Tensor, sample_rate: int):
    """Apply RNNoise denoising."""
    try:
        import pyrnnoise
        from pyrnnoise import Denoiser
        
        # Convert to numpy
        audio_np = audio.cpu().numpy().flatten()
        
        # Initialize RNNoise
        denoiser = Denoiser()
        
        # Process in chunks (RNNoise works on 480-sample frames)
        frame_size = 480
        denoised_chunks = []
        
        for i in range(0, len(audio_np), frame_size):
            chunk = audio_np[i:i+frame_size]
            if len(chunk) == frame_size:
                denoised_chunk = denoiser.process(chunk)
                denoised_chunks.append(denoised_chunk)
            else:
                # Pad last chunk if needed
                chunk_padded = np.pad(chunk, (0, frame_size - len(chunk)))
                denoised_chunk = denoiser.process(chunk_padded)
                denoised_chunks.append(denoised_chunk[:len(chunk)])
        
        denoised_audio = np.concatenate(denoised_chunks)
        return torch.tensor(denoised_audio, dtype=audio.dtype)
    except Exception as e:
        print(f"Warning: RNNoise denoising failed: {e}")
        return audio

def incremental_dns64_denoising(audio: torch.Tensor, sample_rate: int, device: str, max_passes: int = 10, improvement_threshold: float = 0.001):
    """Apply DNS64 incrementally until no more improvement."""
    try:
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
        
        # Load DNS64 model
        model = pretrained.dns64().to(device)
        
        # Convert audio to DNS64 format
        wav = convert_audio(audio, sample_rate, model.sample_rate, model.chin)
        
        current_audio = wav
        previous_noise_level = float('inf')
        
        print(f"  🔄 Starting incremental DNS64 denoising (max {max_passes} passes)...")
        
        for pass_num in range(max_passes):
            # Apply DNS64
            with torch.no_grad():
                denoised = model(current_audio.unsqueeze(0).to(device))[0].cpu()
            
            # Analyze noise level
            noise_analysis = analyze_noise_levels(denoised, model.sample_rate, f"Pass {pass_num + 1}")
            current_noise_level = noise_analysis['noise_floor_95']
            
            # Check improvement
            improvement = previous_noise_level - current_noise_level
            print(f"     Pass {pass_num + 1}: Noise floor = {current_noise_level:.6f}, Improvement = {improvement:.6f}")
            
            # Stop if no significant improvement
            if improvement < improvement_threshold:
                print(f"     ✅ Stopping: No significant improvement ({improvement:.6f} < {improvement_threshold})")
                break
            
            current_audio = denoised
            previous_noise_level = current_noise_level
        
        # Convert back to original sample rate
        if model.sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(model.sample_rate, sample_rate)
            denoised = resampler(denoised)
        
        return denoised
        
    except Exception as e:
        print(f"Warning: Incremental DNS64 denoising failed: {e}")
        return audio

def advanced_vad_denoise_audio(audio_path: Path, device: str, denoise_method: str = "dns64_incremental", 
                              max_passes: int = 10, improvement_threshold: float = 0.001):
    """Advanced VAD-based denoising with adaptive multi-pass logic and detailed analysis."""
    
    print(f"\n🎯 Processing: {audio_path.name}")
    print(f"   Method: {denoise_method}")
    
    # Detect speech segments
    vad_result = detect_speech_segments(audio_path, device)
    if vad_result is None:
        print("  ❌ VAD detection failed, using original audio")
        return None
    
    speech_timestamps, waveform, sample_rate = vad_result
    
    # Analyze original audio
    print(f"  📊 Original Audio Analysis:")
    original_analysis = analyze_noise_levels(waveform, sample_rate, "Original")
    
    if not speech_timestamps:
        print("  ⚠️  No speech detected, applying full denoising")
        if denoise_method == "dns64_incremental":
            denoised = incremental_dns64_denoising(waveform, sample_rate, device, max_passes, improvement_threshold)
        elif denoise_method == "spectral":
            denoised = waveform
            for i in range(max_passes):
                prev_noise = analyze_noise_levels(denoised, sample_rate, f"Spectral Pass {i+1} (no speech)")['noise_floor_95']
                denoised_new = apply_spectral_subtraction(denoised, sample_rate)
                new_noise = analyze_noise_levels(denoised_new, sample_rate, f"Spectral Pass {i+1} (no speech, after)")['noise_floor_95']
                if prev_noise - new_noise < improvement_threshold:
                    print(f"     ✅ Stopping: No significant improvement ({prev_noise - new_noise:.6f} < {improvement_threshold})")
                    break
                denoised = denoised_new
        elif denoise_method == "rnnoise":
            denoised = waveform
            for i in range(max_passes):
                prev_noise = analyze_noise_levels(denoised, sample_rate, f"RNNoise Pass {i+1} (no speech)")['noise_floor_95']
                denoised_new = apply_rnnoise_denoising(denoised, sample_rate)
                new_noise = analyze_noise_levels(denoised_new, sample_rate, f"RNNoise Pass {i+1} (no speech, after)")['noise_floor_95']
                if prev_noise - new_noise < improvement_threshold:
                    print(f"     ✅ Stopping: No significant improvement ({prev_noise - new_noise:.6f} < {improvement_threshold})")
                    break
                denoised = denoised_new
        else:
            denoised = apply_dns64_denoising(audio_path, device, num_passes=1)
        
        if denoised is not None:
            analyze_noise_levels(denoised, sample_rate, "Fully Denoised")
            return denoised
        return waveform.squeeze()
    
    processed_segments = []
    crossfade_samples = int(0.1 * sample_rate)
    
    # Helper for adaptive denoising of non-speech
    def adaptive_denoise(segment, method, label):
        print(f"  🔇 Adaptive denoising for {label} (len={segment.shape[-1]})")
        noise_history = []
        denoised = segment
        for i in range(max_passes):
            prev_noise = analyze_noise_levels(denoised, sample_rate, f"{label} Pass {i+1}")['noise_floor_95']
            noise_history.append(prev_noise)
            if method == "dns64_incremental":
                denoised_new = incremental_dns64_denoising(denoised, sample_rate, device, max_passes=1, improvement_threshold=improvement_threshold)
            elif method == "spectral":
                denoised_new = apply_spectral_subtraction(denoised, sample_rate)
            elif method == "rnnoise":
                denoised_new = apply_rnnoise_denoising(denoised, sample_rate)
            else:
                denoised_new = apply_dns64_denoising(audio_path, device, num_passes=1)
            new_noise = analyze_noise_levels(denoised_new, sample_rate, f"{label} Pass {i+1} (after)")['noise_floor_95']
            if prev_noise - new_noise < improvement_threshold:
                print(f"     ✅ Stopping: No significant improvement ({prev_noise - new_noise:.6f} < {improvement_threshold})")
                break
            denoised = denoised_new
        print(f"  🔇 {label} summary: initial={noise_history[0]:.6f}, final={noise_history[-1]:.6f}, passes={len(noise_history)}")
        return denoised
    
    # Initial silence
    if speech_timestamps[0][0] > 0:
        initial_silence_end = speech_timestamps[0][0]
        if initial_silence_end > crossfade_samples:
            print(f"  🔇 Processing initial silence ({initial_silence_end/sample_rate:.2f}s)")
            initial_silence = waveform[:, :initial_silence_end]
            denoised_silence = adaptive_denoise(initial_silence, denoise_method, "Initial Silence")
            processed_segments.append((0, denoised_silence.squeeze()))
    
    # Speech segments
    for i, (start, end) in enumerate(speech_timestamps):
        print(f"  🗣️  Processing speech segment {i+1} ({start/sample_rate:.2f}s - {end/sample_rate:.2f}s)")
        pad_samples = int(0.1 * sample_rate)
        segment_start = max(0, start - pad_samples)
        segment_end = min(len(waveform[0]), end + pad_samples)
        segment = waveform[:, segment_start:segment_end]
        # Gentle denoising for speech, but print noise analysis
        print(f"  🗣️  Speech segment {i+1} noise before:")
        analyze_noise_levels(segment, sample_rate, f"Speech {i+1} (before)")
        if denoise_method == "dns64_incremental":
            denoised_segment = incremental_dns64_denoising(segment, sample_rate, device, max_passes=2, improvement_threshold=improvement_threshold)
        elif denoise_method == "spectral":
            denoised_segment = apply_spectral_subtraction(segment, sample_rate, noise_reduction_factor=1.0)
        elif denoise_method == "rnnoise":
            denoised_segment = apply_rnnoise_denoising(segment, sample_rate)
        else:
            denoised_segment = apply_dns64_denoising(audio_path, device, num_passes=1)
        print(f"  🗣️  Speech segment {i+1} noise after:")
        analyze_noise_levels(denoised_segment, sample_rate, f"Speech {i+1} (after)")
        if denoised_segment is not None:
            processed_segments.append((segment_start, denoised_segment.squeeze()))
        else:
            processed_segments.append((segment_start, segment.squeeze()))
    
    # Final silence
    last_speech_end = speech_timestamps[-1][1]
    if last_speech_end < len(waveform[0]):
        final_silence_start = last_speech_end
        final_silence = waveform[:, final_silence_start:]
        if len(final_silence[0]) > crossfade_samples:
            print(f"  🔇 Processing final silence ({(len(waveform[0])-final_silence_start)/sample_rate:.2f}s)")
            denoised_silence = adaptive_denoise(final_silence, denoise_method, "Final Silence")
            processed_segments.append((final_silence_start, denoised_silence.squeeze()))
    
    # Reconstruct audio with crossfades
    if not processed_segments:
        return waveform.squeeze()
    
    processed_segments.sort(key=lambda x: x[0])
    result = torch.zeros(len(waveform[0]))
    
    for i, (start, segment) in enumerate(processed_segments):
        end = start + len(segment)
        if i == 0:
            result[:end] = segment
        else:
            prev_end = processed_segments[i-1][0] + len(processed_segments[i-1][1])
            if start > prev_end:
                result[prev_end:start] = 0
                result[start:end] = segment
            else:
                overlap = prev_end - start
                if overlap > 0:
                    crossfade_len = min(overlap, crossfade_samples)
                    result[start:start+crossfade_len] = crossfade_audio(
                        result[start:start+crossfade_len], 
                        segment[:crossfade_len], 
                        crossfade_len
                    )
                    result[start+crossfade_len:end] = segment[crossfade_len:]
                else:
                    result[start:end] = segment
    
    # Final analysis
    print(f"  📊 Final Result Analysis:")
    analyze_noise_levels(result, sample_rate, "Final Denoised")
    
    return result

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

def process_one(src: Path, out_dir: Path, device: str, force: bool, use_vad_denoising: bool = False, use_both: bool = False, 
                denoise_strength_silence: int = 3, denoise_strength_speech: int = 1, target_noise_floor: float = 0.0005,
                denoise_method: str = "dns64_incremental", max_passes: int = 10, improvement_threshold: float = 0.001) -> bool:
    # Extract metadata title from source file
    title = get_metadata_title(src)
    safe_title = ''.join(c for c in title if c.isalnum() or c in ' _-').strip()
    safe_title = safe_title.replace(' ', '_')
    final_out = out_dir / f"{safe_title}.mp3"
    
    if final_out.exists() and not force:
        return False  # skipped

    tmp_root = out_dir / ".tmp"
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td = Path(td)
        tmp_in  = td / "in.wav"
        tmp_out = td / "out.wav"
        tmp_denoised = td / "denoised.wav"
        tmp_final_wav = td / "final.wav"

        # Always convert to uniform wav (cheap & safe)
        to_wav48k_mono(src, tmp_in)

        # Process based on method
        try:
            if use_both:
                # Step 1: Apply advanced VAD-based denoising
                print(f"  Step 1: Advanced denoising {src.name}...")
                denoised_audio = advanced_vad_denoise_audio(tmp_in, device, denoise_method, max_passes, improvement_threshold)
                if denoised_audio is not None:
                    torchaudio.save(str(tmp_denoised), denoised_audio.unsqueeze(0), 48000)
                    
                    # Step 2: Apply AudioSR upscaling to denoised audio
                    print(f"  Step 2: Upscaling {src.name}...")
                    audiosr_enhance(tmp_denoised, tmp_out, device)
                    if tmp_out.exists():
                        shutil.move(tmp_out, tmp_final_wav)
                    else:
                        shutil.move(tmp_denoised, tmp_final_wav)
                else:
                    # Fallback to AudioSR only if advanced denoising fails
                    print(f"  Advanced denoising failed, using AudioSR only for {src.name}...")
                    audiosr_enhance(tmp_in, tmp_out, device)
                    if tmp_out.exists():
                        shutil.move(tmp_out, tmp_final_wav)
                    else:
                        shutil.move(tmp_in, tmp_final_wav)
                        
            elif use_vad_denoising:
                # Apply advanced VAD-based denoising only
                denoised_audio = advanced_vad_denoise_audio(tmp_in, device, denoise_method, max_passes, improvement_threshold)
                if denoised_audio is not None:
                    torchaudio.save(str(tmp_final_wav), denoised_audio.unsqueeze(0), 48000)
                else:
                    # Fallback to AudioSR if advanced denoising fails
                    audiosr_enhance(tmp_in, tmp_out, device)
                    if tmp_out.exists():
                        shutil.move(tmp_out, tmp_final_wav)
                    else:
                        shutil.move(tmp_in, tmp_final_wav)
            else:
                # Use AudioSR enhancement only
                audiosr_enhance(tmp_in, tmp_out, device)
                # If tmp_out exists, use it (CLI fallback). Otherwise, use tmp_in (Python API overwrites input)
                if tmp_out.exists():
                    shutil.move(tmp_out, tmp_final_wav)
                else:
                    shutil.move(tmp_in, tmp_final_wav)
            
            # Convert final WAV to MP3 with metadata
            print(f"  Step 3: Encoding MP3 with metadata for {src.name}...")
            mp3_result = encode_mp3_with_metadata(tmp_final_wav, final_out, title)
            if mp3_result is None:
                print(f"Error: Failed to encode MP3 for {src.name}")
                return False
                
        except Exception as e:
            print(f"Error processing {src}: {e}")
            return False

    return True  # processed

# ────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("Batch AudioSR enhancer → MP3 with metadata (resume‑safe)")
    ap.add_argument("--input_dir", default="raw", help="Source directory")
    ap.add_argument("--output_dir", default="output2", help="Destination directory for MP3 files")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--force", action="store_true", help="Re‑process even if MP3 exists")
    ap.add_argument("--use_vad_denoising", action="store_true", 
                   help="Use VAD-based denoising instead of AudioSR (better for noise reduction)")
    ap.add_argument("--both", action="store_true", 
                   help="Apply VAD denoising first, then AudioSR upscaling (best of both worlds)")
    ap.add_argument("--denoise_strength_silence", type=int, default=3, help="DNS64 passes for silence (default: 3)")
    ap.add_argument("--denoise_strength_speech", type=int, default=1, help="DNS64 passes for speech (default: 1)")
    ap.add_argument("--target_noise_floor", type=float, default=0.0005, help="Target noise floor for silence (default: 0.0005)")
    ap.add_argument("--denoise_method", choices=["dns64_incremental", "spectral", "rnnoise", "dns64"], 
                   default="dns64_incremental", help="Denoising method (default: dns64_incremental)")
    ap.add_argument("--max_passes", type=int, default=10, help="Maximum DNS64 passes (default: 10)")
    ap.add_argument("--improvement_threshold", type=float, default=0.001, help="Improvement threshold for stopping (default: 0.001)")
    
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
    
    # Determine processing method
    if args.both:
        method = f"Advanced {args.denoise_method} + AudioSR Upscaling → MP3"
    elif args.use_vad_denoising:
        method = f"Advanced {args.denoise_method} → MP3"
    else:
        method = "AudioSR Enhancement → MP3"
    
    print(f"🚀 Using device: {device}")
    print(f"🎯 Processing method: {method}")
    print(f"🔧 Denoising method: {args.denoise_method}")
    print(f"📊 Max passes: {args.max_passes}, Improvement threshold: {args.improvement_threshold}")
    print(f"🏷️  Preserving metadata from source files")

    done, skipped = 0, 0
    for f in tqdm(files, desc="Processing"):
        if process_one(f, out_dir, device, args.force, args.use_vad_denoising, args.both, 
                      args.denoise_strength_silence, args.denoise_strength_speech, args.target_noise_floor,
                      args.denoise_method, args.max_passes, args.improvement_threshold):
            done += 1
        else:
            skipped += 1

    print(f"\n✅ Finished. New/updated MP3s: {done}  |  Skipped: {skipped}\nOutputs → {out_dir}\n")


if __name__ == "__main__":
    main()

