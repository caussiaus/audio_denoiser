#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

def convert_to_mp3(input_file, output_file):
    """Convert WAV to high quality MP3 using FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_file),
        "-codec:a", "libmp3lame",
        "-q:a", "0",  # Highest quality VBR setting
        str(output_file)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error during conversion: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="High Quality WAV to MP3 Converter")
    parser.add_argument("--input_dir", default="output", help="Input directory with WAV files")
    parser.add_argument("--output_dir", default="output_mp3", help="Output directory for MP3 files")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of already converted files")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Get all WAV files
    wav_files = list(input_dir.glob("*.wav"))
    
    if not wav_files:
        print("❌ No WAV files found in input directory!")
        return
    
    if not args.force:
        # Skip files that are already converted
        original_count = len(wav_files)
        wav_files = [f for f in wav_files if not (output_dir / f"{f.stem}.mp3").exists()]
        skipped_count = original_count - len(wav_files)
        if skipped_count > 0:
            print(f"⏭️  Skipping {skipped_count} already converted files.")
    
    if not wav_files:
        print("✅ All files already converted!")
        return
    
    print(f"🔄 Converting {len(wav_files)} WAV files to MP3...")
    
    successful = 0
    failed = 0
    
    for wav_file in tqdm(wav_files, desc="Converting files"):
        output_file = output_dir / f"{wav_file.stem}.mp3"
        
        if convert_to_mp3(wav_file, output_file):
            successful += 1
        else:
            failed += 1
            print(f"❌ Failed to convert: {wav_file.name}")
    
    print("\n✅ Conversion complete!")
    print(f"   Successfully converted: {successful}")
    print(f"   Failed: {failed}")

if __name__ == "__main__":
    main()