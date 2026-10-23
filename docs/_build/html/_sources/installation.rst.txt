============
Installation
============

Prerequisites
=============

Before installing IFU Analyzer, you need:

- **Python**: 3.12 only. The pinned versions in ``requirements.txt`` are tested on 3.12.
  Python 3.11 and 3.13 fail at the install step. If you do not have 3.12, use Conda (Step 0 below).
- **Conda** (recommended): Installs Python 3.12 for you, in a separate environment
- **Git** (recommended): To clone the repository and to get updates
- **Operating System**: Windows, macOS, or Linux

Getting Started
================

**Important**: IFU Analyzer requires specific versions of packages to avoid compatibility issues.
Creating a dedicated environment (via conda or ``venv``) is strongly recommended.

Step 0 (Recommended): Create an Isolated Environment
------------------------------------------------------------

Creating a dedicated environment before installing anything keeps IFU Analyzer's
dependencies separate from your system Python. Pick one:

**Option A: Conda** (recommended)

Conda installs Python 3.12 for you. You do not need to change the Python on your computer.

.. code-block:: bash

   conda create -n ifu-analyzer python=3.12
   conda activate ifu-analyzer

**Option B: venv** (Python 3.12 must already be installed; activation differs by platform)

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

Option A: Using Git (recommended, because it lets you get updates)

1. Install Git, if it is not on your computer yet:

   - **macOS:** ``brew install git``
   - **Windows:** download the installer from https://git-scm.com
   - **Linux:** use your package manager, for example ``sudo apt install git``

2. Go to the folder where you want the ``IFU-Analyzer`` folder to be created. Then clone the repository:

   .. code-block:: bash

      git clone https://github.com/ilariapascucci/IFU-Analyzer.git
      cd IFU-Analyzer

   After the clone, you are on the ``main`` branch.

3. To switch to a different branch, run:

   .. code-block:: bash

      git checkout <branch-name>

Option B: Download ZIP file

- Go to https://github.com/ilariapascucci/IFU-Analyzer
- Click "Code" → "Download ZIP"
- Extract the ZIP file
- Open a terminal/command prompt in the extracted folder

A ZIP download cannot update itself. To get a newer version, download the ZIP file again.


2. Install Required Packages
------------------------------

From the IFU-Analyzer folder, run:

.. code-block:: bash

   pip install -r requirements.txt

This installs all dependencies needed to run IFU Analyzer. It can take several minutes.


3. Launch the Application
--------------------------

.. code-block:: bash

   python main.py

A window should open. You're ready to use IFU Analyzer!


Updating IFU Analyzer
=====================

If you cloned the repository with Git, get the latest version each time before you run the program:

.. code-block:: bash

   cd IFU-Analyzer
   git pull

``git pull`` downloads the latest version of your current branch from GitHub.

If ``requirements.txt`` changed, run ``pip install -r requirements.txt`` again.


Using Your Data
===============

IFU Analyzer accepts FITS files from supported instruments:

- **MUSE** (VLT): 3D spectral cubes
- **MIRI** (JWST): Mid-infrared cubes
- **NIRSpec** (JWST): Near-infrared data

To load data:

1. Click **Open FITS** at the top of the window
2. Select your .fits file
3. The application will auto-detect the instrument type

For organizing your data, create a ``data/`` folder in the IFU-Analyzer directory.


Common Issues
=============

Check the FAQ or user guide for help.

**Quick fixes:**

- **Module not found**: Make sure you're in the correct folder and ran ``pip install -r requirements.txt``
- **Import error**: Verify that Python 3.12 is installed: ``python --version``
- **Window won't open**: Check that all packages installed successfully
- **App not reading the file**: Make sure the FITS file is from one of the three supported instruments (MUSE, MIRI, or NIRSpec) and that its header includes the ``INSTRUME`` or ``TELESCOP`` keyword IFU Analyzer uses to auto-detect the instrument. Files without a recognized keyword, or from other instruments, won't load.


Next Steps
==========

1. Start the application: ``python main.py``
2. Explore :doc:`user_guide/index` for features and tutorials
3. Load your first data file!
