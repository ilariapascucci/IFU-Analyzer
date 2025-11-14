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

* Enter the minimum and maximum wavelengths in micrometers (μm):
* Then enter whether you want the data from your local environment or from Hitran.

![image alt](https://github.com/ilariapascucci/IFU-Analyzer/blob/msa-week5/Screenshot%202025-11-14%20042326.png?raw=true)

### Step 3: Enter which species you want to search

* Enter your speci name and also it's isotopologue number.
* After that choose the type of cache you wanna use (2 is recommended).

![image alt](https://github.com/ilariapascucci/IFU-Analyzer/blob/msa-week5/Screenshot%202025-11-14%20051646.png?raw=true)

### Step 4: Analyze your data

* Now if you entered correct data it would give a output like this

![image alt](https://github.com/ilariapascucci/IFU-Analyzer/blob/msa-week5/Screenshot%202025-11-14%20042429.png?raw=true)

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'hapi'`

**Solution:**
```
pip install hitran-api
```
