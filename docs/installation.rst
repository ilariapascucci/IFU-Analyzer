============
Installation
============

Prerequisites
=============

Before installing IFU Analyzer, you need:

- **Python**: 3.8 or higher
- **Git** (optional): For cloning the repository
- **Operating System**: Windows or macOS

Getting Started
===============

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

That's it! This installs all 10 dependencies needed to run IFU Analyzer.


3. Launch the Application
--------------------------

.. code-block:: bash

   python main.py

A window should open. You're ready to use IFU Analyzer!


Using Your Data
===============

IFU Analyzer accepts FITS files from any supported instrument:

- **MUSE** (VLT): 3D spectral cubes
- **ALMA**: Radio/millimeter data
- **MIRI** (JWST): Mid-infrared cubes
- **NIRSpec** (JWST): Near-infrared data
- **Generic**: Any 2D/3D FITS image

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
2. Read :doc:`quickstart` for a 5-minute tutorial
3. Explore :doc:`user_guide/index` for features
4. Load your first data file!
