# 🔬 HITRAN/Local Spectral Line Data Extractor

A Python command-line tool for extracting and analyzing molecular spectroscopic data from the HITRAN database or local line lists.

---

## Purpose

This tool provides astronomers and spectroscopists with a unified interface for querying spectral line data from two sources: the HITRAN molecular spectroscopic database (55+ molecules) and custom local line lists. It enables users to quickly extract spectral lines within specific wavelength ranges, identify molecular transitions by species (with support for exact and partial matching), and retrieve comprehensive line parameters including wavelengths, wavenumbers, Einstein A-coefficients, and upper energy levels. The tool features intelligent caching for offline HITRAN access, making it ideal for observational planning, spectral line identification in astronomical data, and laboratory spectroscopy applications.

---

### Prerequisites

- Python 3.7 or higher
- Git (for cloning the repository)
- pip (Python package manager)

## Requirements

- numpy
- hitran-api

Install all dependencies:

```
pip install numpy hitran-api
```


## Installation

### Clone the Repository

```
git clone https://github.com/ilariapascucci/IFU-Analyzer.git
cd IFU-Analyzer
git checkout msa-week5
```
## How to run the program?

### Step 1: Launch the Tool

Run the Python script:
```
python spectral_line_extractor.py
```

The program will display the welcome screen and prompt you for input.

### Step 2: Enter Wavelength Range

Enter the minimum and maximum wavelengths in micrometers (μm):
