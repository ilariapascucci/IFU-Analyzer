"""
Dialog for setting continuum fitting parameters
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QScrollArea, QWidget, QPushButton, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import threading

from .gui_sizing import fit_to_screen


class ContinuumCubeWorker(QThread):
    """Background worker thread for saving the continuum and continuum-subtracted cubes"""
    finished = pyqtSignal(int)  # Emits the number of pixels with data but no continuum
    error = pyqtSignal(str)  # Emits error message if failed
    progress = pyqtSignal(int, int)  # (current, total) - emits progress updates

    def __init__(self, cube_data, continuum_path, contsub_path, continuum_params=None,
                 anchor_wavelengths=None, n_jobs=8):
        """Build the worker. Call start() to run it.

        Args:
            cube_data (CubeData): Loaded cube to fit and subtract.
            continuum_path (str): Destination FITS path of the continuum cube.
            contsub_path (str): Destination FITS path of the continuum-subtracted cube.
            continuum_params (dict): fit_auto_continuum() keyword arguments, or None for the defaults.
            anchor_wavelengths (array): Anchor wavelengths for the custom fit, or None for the automatic fit.
            n_jobs (int): Number of worker processes, capped by available memory.

        Returns:
            None
        """
        super().__init__()
        self.cube_data = cube_data
        self.continuum_path = continuum_path
        self.contsub_path = contsub_path
        self.continuum_params = continuum_params
        self.anchor_wavelengths = anchor_wavelengths
        self.n_jobs = n_jobs
        self._cancel_event = threading.Event()

    def cancel(self):
        """Request that the run stop after the currently in-flight rows finish.
        Does not block. Nothing is written when a run is cancelled.

        Args:
            None

        Returns:
            None
        """
        self._cancel_event.set()

    def run(self):
        """Fit the continuum and write both cubes in the background thread.

        Args:
            None

        Returns:
            None
        """
        try:
            from .continuum_cube import export_continuum_cubes
            n_failed = export_continuum_cubes(
                self.cube_data, self.continuum_path, self.contsub_path,
                continuum_params=self.continuum_params, anchor_wavelengths=self.anchor_wavelengths,
                n_jobs=self.n_jobs,
                progress_callback=self.progress.emit, cancel_event=self._cancel_event,
            )
            # n_failed is None when cancel() fired before completion; nothing is listening by then.
            if n_failed is not None:
                self.finished.emit(n_failed)
        except Exception as e:
            self.error.emit(str(e))


class ContinuumParametersDialog(QDialog):
    """Dialog to set continuum fitting parameters with explanations"""
    
    def __init__(self, parent=None):
        """Build the continuum parameters dialog with default values.

        Args:
            parent (QWidget): Parent widget, or None for a top-level dialog.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Continuum Fitting Parameters")
        self.setMinimumSize(480, 400)
        fit_to_screen(self, 650, 600)
        
        # Default values from current implementation
        self.params = {
            'window_size': 51,
            'sigma_threshold': 2.0,
            'peak_width': 1,
            'peak_distance': 3,
            'peak_exclusion': 3,
            'iterations': 2,
            'smoothing_factor': 5
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Create the dialog UI with parameter inputs and explanations.

        Args:
            None

        Returns:
            None
        """
        layout = QVBoxLayout()
        
        # Title and algorithm explanation
        title_label = QLabel("<h2>Automatic Continuum Fitting</h2>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Algorithm description
        algo_text = QTextEdit()
        algo_text.setReadOnly(True)
        algo_text.setMaximumHeight(150)
        algo_text.setHtml("""
        <b>Algorithm Overview:</b><br>
        This tool automatically fits the continuum baseline of your spectrum using a 4-step process:<br>
        <ol>
        <li><b>Median Filtering</b> - Smooths noise while preserving baseline trends</li>
        <li><b>Peak Detection & Masking</b> - Identifies and excludes emission/absorption lines</li>
        <li><b>Iterative Sigma Clipping</b> - Removes outliers and refines the continuum fit</li>
        <li><b>Spline Smoothing</b> - Creates a smooth, continuous continuum curve</li>
        </ol>
        Adjust the parameters below to optimize the fit for your data.
        """)
        layout.addWidget(algo_text)
        
        # Separator
        layout.addWidget(QLabel("<hr>"))
        
        # Create scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        params_layout = QVBoxLayout(scroll_widget)
        
        # Dictionary to store input widgets
        self.input_widgets = {}
        
        # PARAMETER 1: Median Filter Window Size
        params_layout.addWidget(self._create_section_header("1. Median Filter Window Size"))
        self.input_widgets['window_size'] = self._create_parameter_input(
            params_layout,
            name='window_size',
            label='Window Size (pixels):',
            default=51,
            min_val=3,
            max_val=201,
            step=2,
            description=(
                "<b>What it does:</b> Sets the size of the sliding window for median filtering.<br>"
                "<b>Effect:</b> Larger values = smoother baseline, but may miss fine structure.<br>"
                "<b>Typical range:</b> 21-101 pixels<br>"
                "<b>Recommendation:</b> Use 51 for typical spectra, 101 for very noisy data<br>"
                "<i>Note: Must be an odd number</i>"
            )
        )
        
        # PARAMETER 2: Sigma Threshold for Peak Detection
        params_layout.addWidget(self._create_section_header("2. Peak Detection Threshold"))
        self.input_widgets['sigma_threshold'] = self._create_parameter_input(
            params_layout,
            name='sigma_threshold',
            label='Sigma Threshold (σ):',
            default=2.0,
            min_val=0.5,
            max_val=5.0,
            step=0.1,
            description=(
                "<b>What it does:</b> Sets the detection threshold for emission/absorption lines.<br>"
                "<b>Effect:</b> Lower values = more sensitive (detects weaker lines)<br>"
                "<b>Typical range:</b> 1.5-3.0 σ<br>"
                "<b>Recommendation:</b> Use 2.0 for normal data, 3.0 for noisy data, 1.5 for clean data"
            )
        )
        
        # PARAMETER 3: Minimum Peak Width
        params_layout.addWidget(self._create_section_header("3. Minimum Peak Width"))
        self.input_widgets['peak_width'] = self._create_parameter_input(
            params_layout,
            name='peak_width',
            label='Minimum Width (pixels):',
            default=1,
            min_val=1,
            max_val=20,
            step=1,
            description=(
                "<b>What it does:</b> Minimum width required for a feature to be considered a peak.<br>"
                "<b>Effect:</b> Filters out noise spikes and very narrow artifacts<br>"
                "<b>Typical range:</b> 1-5 pixels<br>"
                "<b>Recommendation:</b> Use 1 for narrow lines, 3-5 for broader features"
            )
        )
        
        # PARAMETER 4: Minimum Peak Distance
        params_layout.addWidget(self._create_section_header("4. Minimum Peak Separation"))
        self.input_widgets['peak_distance'] = self._create_parameter_input(
            params_layout,
            name='peak_distance',
            label='Minimum Distance (pixels):',
            default=3,
            min_val=1,
            max_val=50,
            step=1,
            description=(
                "<b>What it does:</b> Minimum separation between detected peaks.<br>"
                "<b>Effect:</b> Prevents detection of multiple peaks in the same feature<br>"
                "<b>Typical range:</b> 3-10 pixels<br>"
                "<b>Recommendation:</b> Use 3 for densely-packed lines, 10 for sparse spectra"
            )
        )
        
        # PARAMETER 5: Peak Exclusion Zone
        params_layout.addWidget(self._create_section_header("5. Peak Exclusion Zone"))
        self.input_widgets['peak_exclusion'] = self._create_parameter_input(
            params_layout,
            name='peak_exclusion',
            label='Exclusion Radius (pixels):',
            default=3,
            min_val=0,
            max_val=20,
            step=1,
            description=(
                "<b>What it does:</b> Number of pixels on each side of a peak to exclude from continuum fit.<br>"
                "<b>Effect:</b> Excludes the wings of emission/absorption lines<br>"
                "<b>Typical range:</b> 2-10 pixels<br>"
                "<b>Recommendation:</b> Use 3 for narrow lines, 5-10 for broad lines"
            )
        )
        
        # PARAMETER 6: Sigma Clipping Iterations
        params_layout.addWidget(self._create_section_header("6. Outlier Removal Iterations"))
        self.input_widgets['iterations'] = self._create_parameter_input(
            params_layout,
            name='iterations',
            label='Number of Iterations:',
            default=2,
            min_val=0,
            max_val=5,
            step=1,
            description=(
                "<b>What it does:</b> Number of times to iteratively remove outliers above the continuum.<br>"
                "<b>Effect:</b> More iterations = cleaner continuum but slower processing<br>"
                "<b>Typical range:</b> 1-3 iterations<br>"
                "<b>Recommendation:</b> Use 2 for most cases, 3 for complex spectra with many lines"
            )
        )
        
        # PARAMETER 7: Spline Smoothing Factor
        params_layout.addWidget(self._create_section_header("7. Spline Smoothing Factor"))
        self.input_widgets['smoothing_factor'] = self._create_parameter_input(
            params_layout,
            name='smoothing_factor',
            label='Smoothing Factor:',
            default=5,
            min_val=1,
            max_val=50,
            step=1,
            description=(
                "<b>What it does:</b> Controls the smoothness of the final spline fit.<br>"
                "<b>Effect:</b> Lower values = closer fit to data, higher = smoother curve<br>"
                "<b>Typical range:</b> 3-20<br>"
                "<b>Recommendation:</b> Use 5 for typical data, 10-20 for very noisy spectra<br>"
                "<i>Formula: s = N / smoothing_factor (N = number of data points)</i>"
            )
        )
        
        # Finalize scroll area
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Separator
        layout.addWidget(QLabel("<hr>"))
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("Apply & Fit Continuum")
        ok_btn.setStyleSheet("font-weight: bold; background-color: #4a90e2; color: white;")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _create_section_header(self, text):
        """Create a styled section header.

        Args:
            text (str): Header text to display.

        Returns:
            label (QLabel): The styled header label.
        """
        label = QLabel(f"<b style='font-size: 11pt; color: #2d73c5;'>{text}</b>")
        return label

    def _create_parameter_input(self, parent_layout, name, label, default, min_val, max_val, step, description):
        """Create a parameter input widget with label and description.

        Args:
            parent_layout (QLayout): Layout to add the widget to.
            name (str): Parameter key, e.g. 'window_size'.
            label (str): Label text shown next to the input.
            default (int or float): Initial value.
            min_val (int or float): Minimum allowed value.
            max_val (int or float): Maximum allowed value.
            step (int or float): Spin box step size.
            description (str): HTML description shown below the input.

        Returns:
            spinbox (QSpinBox or QDoubleSpinBox): The created input widget.
        """
        # Container for this parameter
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        param_layout.setContentsMargins(10, 5, 10, 10)
        
        # Input row
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel(label))
        
        # Choose SpinBox type based on whether we need decimals
        if isinstance(default, float) or step < 1:
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(1)
        else:
            spinbox = QSpinBox()
        
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setSingleStep(step)
        spinbox.setValue(default)
        spinbox.setMinimumWidth(100)
        
        # Special handling for window_size (must be odd)
        if name == 'window_size':
            spinbox.valueChanged.connect(lambda val: self._ensure_odd(spinbox))
        
        input_layout.addWidget(spinbox)
        input_layout.addStretch()
        param_layout.addLayout(input_layout)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("padding: 5px; background-color: #f5f5f5; border-radius: 3px; font-size: 9pt;")
        param_layout.addWidget(desc_label)
        
        parent_layout.addWidget(param_widget)
        
        return spinbox
    
    def _ensure_odd(self, spinbox):
        """Ensure window_size is always odd.

        Args:
            spinbox (QSpinBox): The window_size spin box to correct.

        Returns:
            None
        """
        value = spinbox.value()
        if value % 2 == 0:
            spinbox.setValue(value + 1)
    
    def _reset_defaults(self):
        """Reset all parameters to default values.

        Args:
            None

        Returns:
            None
        """
        defaults = {
            'window_size': 51,
            'sigma_threshold': 2.0,
            'peak_width': 1,
            'peak_distance': 3,
            'peak_exclusion': 3,
            'iterations': 2,
            'smoothing_factor': 5
        }
        
        for param_name, default_value in defaults.items():
            self.input_widgets[param_name].setValue(default_value)
    
    def get_parameters(self):
        """Get the current parameter values.

        Args:
            None

        Returns:
            params (dict): Current continuum-fitting parameter values.
        """
        return {
            'window_size': int(self.input_widgets['window_size'].value()),
            'sigma_threshold': float(self.input_widgets['sigma_threshold'].value()),
            'peak_width': int(self.input_widgets['peak_width'].value()),
            'peak_distance': int(self.input_widgets['peak_distance'].value()),
            'peak_exclusion': int(self.input_widgets['peak_exclusion'].value()),
            'iterations': int(self.input_widgets['iterations'].value()),
            'smoothing_factor': int(self.input_widgets['smoothing_factor'].value())
        }
