============
Installation
============

Prerequisites
=============

Before installing IFU Analyzer, ensure your system meets these requirements:

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 4 GB minimum (8 GB or more recommended for large data cubes)
- **Disk Space**: At least 500 MB for installation and sample data

System-Specific Notes
---------------------

**Windows**: Administrator privileges may be needed for the virtual environment setup.

**macOS**: You may need to install Xcode Command Line Tools:

.. code-block:: bash

   xcode-select --install

**Linux**: Most distributions have Python pre-installed. If not, use your package manager:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install python3 python3-pip python3-venv

   # Fedora/CentOS
   sudo dnf install python3 python3-pip

Dependencies
============

The following Python packages are required:

- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **astropy** - FITS file handling and astronomy utilities
- **PyQt5** - Graphical user interface
- **scipy** - Scientific computing (for fitting algorithms)

Installation Steps
==================

1. Clone or Download the Repository
-----------------------------------

Using Git:

.. code-block:: bash

   git clone https://github.com/ilariapascucci/IFU-Analyzer.git
   cd IFU-Analyzer

Or download as a ZIP file from the project repository and extract it.

2. Create a Virtual Environment (Recommended)
----------------------------------------------

Creating a virtual environment isolates IFU Analyzer's dependencies from your system Python:

**Using venv (included with Python):**

.. code-block:: bash

   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

**Using Conda (if you have Anaconda/Miniconda installed):**

.. code-block:: bash

   conda create -n ifu-analyzer python=3.10
   conda activate ifu-analyzer

3. Install Required Packages
-----------------------------

.. code-block:: bash

   pip install numpy matplotlib astropy PyQt5 scipy

Or install directly from requirements (if available):

.. code-block:: bash

   pip install -r requirements.txt

4. Verify Installation
----------------------

To verify that everything is installed correctly:

.. code-block:: bash

   python main.py

The IFU Analyzer window should open (1100×800 pixels).

Troubleshooting Installation
============================

**Issue: "No module named 'PyQt5'"**

This typically means PyQt5 wasn't installed. Try:

.. code-block:: bash

   pip install --upgrade PyQt5

**Issue: "No module named 'astropy'"**

Install the astropy package:

.. code-block:: bash

   pip install astropy

**Issue: Command not found: python3**

Ensure Python 3.8+ is installed and added to your PATH. Check your Python installation:

.. code-block:: bash

   python --version
   # or
   python3 --version

**Issue: GUI window opens but crashes immediately**

This may indicate a PyQt5 display issue. Try reinstalling PyQt5:

.. code-block:: bash

   pip uninstall PyQt5
   pip install PyQt5

**Issue: "Permission denied" on macOS/Linux**

You may need to use ``sudo`` or fix ownership:

.. code-block:: bash

   # Option 1: Use sudo
   sudo pip install numpy matplotlib astropy PyQt5 scipy

   # Option 2: Fix pip permissions (better practice)
   python -m pip install --user numpy matplotlib astropy PyQt5 scipy

Data Files
==========

IFU Analyzer does not include sample data files in the repository. To use the application:

1. Prepare your own FITS data cubes from supported instruments:
   - MUSE: 3D spectral cubes (.fits)
   - ALMA: Radio interferometry cubes (.fits)
   - MIRI: Mid-infrared spectral cubes (.fits)
   - NIRSpec: Near-infrared spectral cubes (.fits)
   - Generic: Any 2D/3D FITS images

2. Use the "Open FITS" button in the GUI to load your data files

3. Recommended: Create a ``data/`` subdirectory in your IFU-Analyzer folder to organize your datasets

Configuration Files
====================

Optional Configuration
----------------------

Line catalogs and other data files are stored in:

- ``hitran_local/`` - HITRAN spectral line database files
- ``data/`` - Sample and user data

These directories are created automatically when needed.

Next Steps
==========

After successful installation:

1. Start the application: ``python main.py``
2. Read the :doc:`quickstart` guide for basic usage
3. Explore the :doc:`user_guide/index` for detailed features
4. Load your first FITS file!

Getting Help
============

If you encounter issues:

1. Check the :doc:`troubleshooting` section
2. Review the :doc:`faq` for common questions
3. Check the project's issue tracker on GitHub
4. Contact the development team

System Recommendations
======================

For optimal performance, especially with large data cubes:

- **Processor**: Multi-core processor (4+ cores recommended)
- **RAM**: 8-16 GB for cubes with millions of voxels
- **GPU**: Not required, but CPU-based analysis is used
- **Display**: 1920×1080 or higher resolution recommended
- **Network**: Internet connection needed for line catalog updates (optional)

Platform-Specific Tips
======================

**Windows**

- If using Anaconda, always run commands from the "Anaconda Prompt"
- Visual C++ Runtime may need updating for some packages

**macOS**

- M1/M2 Macs: You may need to install some packages with: ``pip install --upgrade --force-reinstall <package_name>``
- Homebrew installation can help: ``brew install python3``

**Linux**

- Ensure you have the development headers: ``python3-dev`` package
- Many distributions require: ``sudo apt-get install python3-tk`` for Tkinter support
