# IFU Analyzer

An interactive Python tool for visualizing and analyzing Integral Field Unit (IFU) spectroscopic data from various astronomical instruments. This application provides a user-friendly interface for exploring 3D spectral data cubes, analyzing spectra, and performing region-based analysis.

## Features

- Support for multiple instruments:
  - MUSE (VLT)
  - ALMA
  - MIRI (JWST)
  - NIRSpec (JWST)
- Interactive visualization:
  - Full spectrum display
  - Dual slice view for wavelength/velocity comparison
  - Real-time spectrum updates
  - Region selection tools (point, rectangle, circle, ellipse, custom)
- Analysis capabilities:
  - Coordinate conversion (pixel to RA/DEC)
  - Region-based spectral extraction
  - Statistical analysis of selected regions
  - Memory-efficient handling of large data cubes

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

```bash
# Install required Python packages
pip install numpy matplotlib astropy PyQt5
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ilariapascucci/IFU-Analyzer.git
cd IFU-Analyzer
```

2. (Optional) Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

### Data Files

The program works with FITS format data cubes from various instruments. You can open FITS files from any location on your computer using the GUI's "Open FITS" button. Supported data types include:
- MUSE: 3D spectral cubes (.fits)
- ALMA: Radio interferometry cubes (.fits)
- MIRI: Mid-infrared spectral cubes (.fits)
- NIRSpec: Near-infrared spectral cubes (.fits)

Note: FITS files are not included in the repository. You'll need to provide your own data files from your observations or archives.

## Usage

1. Start the application:
```bash
python main.py
```

2. Using the interface:
   - Click "Open FITS" to select a data cube
   - Click "Load Cube" to load the selected file
   - Use the spectrum plot (bottom) to select wavelength/velocity slices
   - Use shape buttons on the sides to select regions
   - Click points or drag regions to analyze specific areas
   - Statistics update automatically for selections

### Data Requirements

- FITS files must be 3D data cubes
- Supported formats:
  - MUSE: Primary extension or extension 1
  - ALMA: Primary extension
  - MIRI/NIRSpec: 'SCI' extension

## File Structure

```
IFU-Analyzer/
├── main.py              # Application entry point
├── cube_explorer/
│   ├── __init__.py
│   ├── gui.py          # Main GUI implementation
│   ├── data_loader.py  # FITS file handling
│   ├── visualization.py # Plotting functions
│   ├── region_selector.py # Region selection tools
│   └── utils.py        # Utility functions
```

## Troubleshooting

Common issues and solutions:

1. Memory Errors:
   - Try closing other applications
   - Use smaller data cubes
   - Application automatically uses memory mapping for large files

2. Display Issues:
   - Ensure PyQt5 is properly installed
   - Update graphics drivers if experiencing rendering problems

3. File Loading Errors:
   - Verify file format matches supported instruments
   - Check FITS header for required keywords
   - Ensure file is not corrupted

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## Version History

- Week 1: Initial GUI implementation, basic FITS loading
- Week 2: Added dual slice view, coordinate display
- Week 3: Implemented region selection, improved layout
- Week 4: Enhanced performance, live updates, layout improvements
- Week 5: LST files integration
- Week 6: Hitran implementation
- Week 7: Added scaling and color options, troubleshooting
- Week 8: Perform Gaussian fits
- Week 9: Export FITS files