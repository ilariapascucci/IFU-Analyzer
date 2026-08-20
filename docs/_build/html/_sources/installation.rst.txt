============
Installation
============

Prerequisites
=============

Before installing IFU Analyzer, you need:

- **Python**: 3.11 or higher (required by the pinned ``astropy`` version)
- **Conda** (optional): For managing Python environments
- **Git** (optional): For cloning the repository
- **Operating System**: Windows, macOS, or Linux

Getting Started
================

**Important**: IFU Analyzer requires specific versions of packages to avoid compatibility issues.
Creating a dedicated environment (via conda or ``venv``) is strongly recommended.

Step 0 (Optional): Create an Isolated Environment
----------------------------------------------------

Creating a dedicated environment before installing anything keeps IFU Analyzer's
dependencies separate from your system Python. Pick one:

**Option A: Conda**

.. code-block:: bash

   conda create -n ifu-analyzer python=3.11
   conda activate ifu-analyzer

**Option B: venv** (activation differs by platform)

*Windows:*

.. code-block:: bat

   python -m venv venv
   venv\Scripts\activate

*macOS / Linux:*

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate


1. Download the Repository
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


2. Install Required Packages
------------------------------

From the IFU-Analyzer folder, run:

.. code-block:: bash

   pip install -r requirements.txt

That's it! This installs all dependencies needed to run IFU Analyzer.


3. Launch the Application
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
- **Import error**: Verify Python 3.11+ is installed: ``python --version``
- **Window won't open**: Check that all packages installed successfully
- **App not reading the file**: Make sure the FITS file is from one of the three supported instruments (MUSE, MIRI, or NIRSpec) and that its header includes the ``INSTRUME`` or ``TELESCOP`` keyword IFU Analyzer uses to auto-detect the instrument. Files without a recognized keyword, or from other instruments, won't load.


Next Steps
==========

1. Start the application: ``python main.py``
2. Explore :doc:`user_guide/index` for features and tutorials
3. Load your first data file!
