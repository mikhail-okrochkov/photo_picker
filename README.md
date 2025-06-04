# Photo Picker

Photo Picker is a Python tool designed to analyze and rank aviation photos, helping photographers select their best shots. It uses various metrics to evaluate photo quality and automatically identifies sequences of similar photos for comparison.

## Features

- **Photo Quality Assessment**
  - Subject focus detection
  - Exposure analysis
  - Sky percentage calculation
  - Halo detection
  - Airline identification with confidence scoring

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
- ultralytics (YOLO)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/photo_judge.git
cd photo_judge
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the YOLO model for airline detection:
```bash
# The model will be downloaded automatically on first run
```

## Usage

### Basic Usage

```python
from src.photo_picker import PhotoPicker

# Initialize the photo picker
picker = PhotoPicker(
    root_dir="/path/to/photos",
    output_excel="photo_assessment.xlsx",
    debug_mode=False  # Set to True for testing with fewer photos
)

# Process all photos
picker.process_all_photos()
```

### Configuration Options

The `PhotoPicker` class accepts the following parameters:

- `root_dir`: Path to the root directory containing photos
- `output_excel`: Path to the output Excel file
- `debug_mode`: Boolean flag for debug mode (processes fewer photos)
- `processed_dirs_file`: Path to JSON file tracking processed directories
- `min_sequence_size`: Minimum number of photos to form a sequence (default: 3)
- `max_sequence_size`: Maximum number of photos in a sequence (default: 10)
- `similarity_threshold`: Threshold for considering photos similar (default: 0.85)

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
   - Detects sky percentage using color analysis
   - Identifies halos around aircraft
   - Uses YOLO model to detect and identify airlines

2. **Sequence Detection**
   - Groups photos taken within 2 seconds of each other
   - Calculates similarity between photos using feature matching
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

- YOLO model for airline detection
- OpenCV for image processing
- scikit-image for advanced image analysis