"""
ColorMap Selection Dialog
Provides a GUI dialog for selecting matplotlib colormaps including colorblind-friendly options.
"""

import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ColorMapDialog(QDialog):
    """Dialog to select a matplotlib colormap (includes colorblind-friendly options)."""
    
    def __init__(self, parent=None):
        """Build the colormap picker with a live gradient preview.

        Args:
            parent (QWidget): Parent widget, or None for a top-level dialog.

        Returns:
            None
        """
        super().__init__(parent)
        self.setWindowTitle("Select Colormap")
        self.setMinimumWidth(300)
        self.selected_cmap = None

        layout = QVBoxLayout()
        info = QLabel("Choose a color scale (preview updates below).")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Colormap list (prioritize perceptually uniform & colorblind-friendly)
        self.cmaps = [
            'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'gray', 'cubehelix',
            'turbo', 'twilight', 'twilight_shifted', 'Spectral', 'RdBu', 'coolwarm'
        ]

        self.list_widget = QListWidget()
        for cm in self.cmaps:
            item = QListWidgetItem(cm)
            self.list_widget.addItem(item)
        self.list_widget.currentTextChanged.connect(self._update_preview)
        layout.addWidget(self.list_widget)

        # Preview matplotlib figure
        self.preview_fig = Figure(figsize=(3, 0.6))
        self.preview_canvas = FigureCanvas(self.preview_fig)
        layout.addWidget(self.preview_canvas)
        self._draw_preview('magma')

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _draw_preview(self, cmap_name):
        """Draw a gradient preview of the selected colormap.

        Args:
            cmap_name (str): Name of the matplotlib colormap to preview.

        Returns:
            None
        """
        self.preview_fig.clear()
        ax = self.preview_fig.add_subplot(111)
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect='auto', cmap=cmap_name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(cmap_name, fontsize=8)
        self.preview_canvas.draw_idle()

    def _update_preview(self, cmap_name):
        """Update preview when user selects a different colormap.

        Args:
            cmap_name (str): Name of the newly selected colormap, or empty.

        Returns:
            None
        """
        if cmap_name:
            self.selected_cmap = cmap_name
            self._draw_preview(cmap_name)

    def accept(self):
        """Handle OK button - ensure a colormap is selected.

        Args:
            None

        Returns:
            None
        """
        if self.selected_cmap is None:
            # Ensure something selected
            current = self.list_widget.currentItem()
            self.selected_cmap = current.text() if current else 'magma'
        super().accept()
