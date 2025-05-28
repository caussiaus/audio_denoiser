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
git clone https://github.com/caussiaus/audio_denoiser.git
cd audio_denoiser
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
python scripts/audio_pipeline.py \
    --input_dir audios \
    --output_dir output \
    --sr 16000 \
    --chunk_secs 30 \
    --overlap_secs 1
```

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
audio_denoiser/
├── audios/           # Input audio files
├── output/           # Enhanced output files
├── scripts/          # Pipeline scripts
│   └── audio_pipeline.py    # Main pipeline script
├── environment.yml   # Conda environment specification
└── README.md         # This file
```

## How It Works

The pipeline processes audio files in three main steps:
1. **Loading**: Reads and preprocesses audio files
2. **Enhancement**: Processes the audio using the DNS64 model in memory-efficient chunks
3. **Saving**: Saves enhanced files with original metadata preserved

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