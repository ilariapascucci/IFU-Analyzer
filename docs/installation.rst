============
Installation
============

Prerequisites
=============

Before installing IFU Analyzer, you need:

- **Python**: 3.8 or higher
- **Conda**: For managing Python environments (optional but recommended)
- **Git** (optional): For cloning the repository
- **Operating System**: Windows or macOS

Getting Started
===============

**Important**: IFU Analyzer requires specific versions of packages to avoid compatibility issues. 
Creating a dedicated conda environment is strongly recommended.

1. Create a Conda Environment (Recommended)
--------------------------------------------

Before proceeding, create a new conda environment for IFU Analyzer:

.. code-block:: bash

   conda create -n ifu-analyzer python=3.10
   conda activate ifu-analyzer

This isolates IFU Analyzer's dependencies from your system Python, preventing package conflicts.

**Note**: You can also use standard Python virtual environments (``venv``), but conda is 
recommended for better package compatibility management.


2. Download the Repository
---------------------------

Option A: Using Git

.. code-block:: bash

   git clone https://github.com/ilariapascucci/IFU-Analyzer.git
   cd IFU-Analyzer

Option B: Download ZIP file

- Go to https://github.com/ilariapascucci/IFU-Analyzer
- Click "Code" → "Download ZIP"
- Extract the ZIP file
- Open a terminal/command prompt in the extracted folder


3. Install Required Packages
------------------------------

From the IFU-Analyzer folder, run:

.. code-block:: bash

   pip install -r requirements.txt

That's it! This installs all dependencies needed to run IFU Analyzer.


4. Launch the Application
--------------------------

.. code-block:: bash

   python main.py

A window should open. You're ready to use IFU Analyzer!


Using Your Data
===============

IFU Analyzer accepts FITS files from supported instruments:

- **MUSE** (VLT): 3D spectral cubes
- **MIRI** (JWST): Mid-infrared cubes
- **NIRSpec** (JWST): Near-infrared data

To load data:

1. Click "File" → "Open FITS" in the application
2. Select your .fits file
3. The application will auto-detect the instrument type

For organizing your data, create a ``data/`` folder in the IFU-Analyzer directory.


Common Issues
=============

Check the FAQ or user guide for help.

**Quick fixes:**

- **Module not found**: Make sure you're in the correct folder and ran ``pip install -r requirements.txt``
- **Import error**: Verify Python 3.8+ is installed: ``python --version``
- **Window won't open**: Check that all packages installed successfully


Next Steps
==========

1. Start the application: ``python main.py``
2. Explore :doc:`user_guide/index` for features and tutorials
3. Load your first data file!
