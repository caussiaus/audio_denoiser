# Audio Enhancement Pipeline

A Python-based pipeline for enhancing audio quality using Facebook's DNS64 model. This tool processes audio files in batches, improving their quality while maintaining metadata and file organization.

## Features

- Batch processing of audio files
- High-quality audio enhancement using DNS64 model
- Maintains original file metadata and naming
- Supports multiple audio formats (WAV, MP3, FLAC)
- Memory-efficient processing with chunked audio handling
- GPU acceleration support (CUDA)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/caseyjussaume/audio_enhancement_pipeline.git
cd audio_enhancement_pipeline
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate audio_env
```

## Usage

### Basic Usage

Place your audio files in the `audios/` directory and run:

```bash
# Make sure you're in the project root directory
python scripts/audio_pipeline.py --input_dir audios --output_dir output
```

Note: Always run the script from the project root directory, not from inside the `scripts/` directory.

### Quick Start

1. Put your audio files in the `audios/` folder
2. Run the pipeline:
```bash
python scripts/audio_pipeline.py --input_dir audios --output_dir output
```

### Parameters

- `--input_dir`: Directory containing input audio files
- `--output_dir`: Directory for enhanced output files
- `--sr`: Sample rate (default: 16000)
- `--chunk_secs`: Size of audio chunks in seconds (default: 30)
- `--overlap_secs`: Overlap between chunks in seconds (default: 1)

### Supported Audio Formats

- WAV
- MP3
- FLAC
- M4A

## Project Structure

```
audio_enhancement_pipeline/
├── audios/           # Input audio files
├── output/           # Enhanced output files
├── scripts/          # Pipeline scripts
│   ├── audio_pipeline.py    # Main pipeline script
│   ├── stitch.py            # Audio stitching utility
│   ├── enhance.py           # Audio enhancement script
│   └── unstitch.py          # Audio unstitching utility
├── environment.yml   # Conda environment specification
└── README.md         # This file
```

## How It Works

1. **Stitching**: Combines multiple audio files into a single file for batch processing
2. **Enhancement**: Processes the combined audio using the DNS64 model
3. **Unstitching**: Splits the enhanced audio back into individual files
4. **Metadata**: Preserves original file names and metadata

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- FFmpeg

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Facebook's DNS64 model for audio enhancement
- PyTorch and TorchAudio for audio processing
- FFmpeg for audio file handling 