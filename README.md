# Photo Picker

Photo Picker is a Python tool designed to analyze and rank aviation photos, helping photographers select their best shots. It uses various metrics to evaluate photo quality and automatically identifies sequences of similar photos for comparison.

## Features

- **Photo Quality Assessment**
  - Subject focus detection using Segment Anything Model (SAM)
  - Exposure analysis
  - Sky percentage calculation
  - Halo detection
  - Airline identification using CLIP

- **Sequence Detection**
  - Automatically groups similar photos into sequences
  - Ranks photos within each sequence based on quality metrics
  - Highlights the best photo in each sequence

- **Excel Report Generation**
  - Creates detailed Excel reports organized by year
  - Includes all quality metrics and assessment data
  - Highlights rank 1 photos in green
  - Supports appending new photos to existing reports

- **Directory Processing**
  - Processes entire directories of photos
  - Tracks processed directories to avoid duplicates
  - Supports both single directory and recursive directory processing

## Requirements

- Python 3.8 or higher
- OpenCV (cv2)
- NumPy
- scikit-image
- openpyxl
- Pillow (PIL)
- PyTorch
- Segment Anything Model (SAM)
- CLIP
- rawpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/photo_picker.git
cd photo_picker
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the required models:
```bash
# SAM model will be downloaded automatically on first run
# CLIP model will be downloaded automatically on first run
```

## Usage

### Basic Usage

```python
from src.photo_picker import PhotoPicker

# Initialize the photo picker
picker = PhotoPicker(
    config_path="config.yaml",  # Path to your config file
    debug_mode=False,  # Set to True for testing with fewer photos
    count_only=False,  # Set to True to only count photos
    force_reprocess=False,  # Set to True to reprocess all directories
    log_level="INFO"  # Set logging level
)

# Process all photos
picker.run()
```

### Configuration File

Create a YAML or JSON configuration file with the following structure:

```yaml
root_directory: "/path/to/your/photos"
```

### Output

The tool generates an Excel file with the following information for each photo:

- Filename and directory
- Date and time
- Airline identification with confidence
- Sky percentage
- Focus score
- Exposure score
- Halo score
- Sequence number (if part of a sequence)
- Rank within sequence (if part of a sequence)

Photos are organized by year in separate tabs, with rank 1 photos highlighted in green.

## How It Works

1. **Photo Quality Assessment**
   - Analyzes each photo for focus, exposure, and composition
   - Uses SAM to detect and segment the main subject
   - Detects sky percentage using color analysis
   - Identifies halos around aircraft
   - Uses CLIP model to detect and identify airlines

2. **Sequence Detection**
   - Groups photos taken within 30 seconds of each other
   - Calculates similarity between photos using histogram comparison
   - Forms sequences of similar photos
   - Ranks photos within each sequence based on quality metrics

3. **Report Generation**
   - Creates Excel reports organized by year
   - Includes all assessment metrics
   - Highlights the best photo in each sequence
   - Supports incremental updates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Segment Anything Model (SAM) for subject detection
- CLIP for airline identification
- OpenCV for image processing
- scikit-image for advanced image analysis