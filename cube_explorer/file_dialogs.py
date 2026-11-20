"""
File dialogs that open in the folder that the user used last.

Every open and save dialog of the application uses these functions. They share one folder,
and they keep it after the application closes.
"""
import os

import matplotlib
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QFileDialog

_SETTINGS_KEY = "last_directory"


def _settings():
    """Give the stored settings of the application.

    Args:
        None

    Returns:
        settings (QSettings): The settings of the user for this application.
    """
    return QSettings("IFU Analyzer", "IFU Analyzer")


def last_directory():
    """Give the folder that the user used last.
    A folder that does not exist any more gives an empty text, so the dialog uses its own default.

    Args:
        None

    Returns:
        folder (str): Path of the last folder, or an empty text if there is none.
    """
    folder = _settings().value(_SETTINGS_KEY, "", type=str)
    return folder if folder and os.path.isdir(folder) else ""


def _remember(path):
    """Store the folder of a file that the user chose. The save button of the plot toolbars
    starts in the same folder.

    Args:
        path (str): Path of the file that the user chose.

    Returns:
        None
    """
    folder = os.path.dirname(os.path.abspath(path))
    _settings().setValue(_SETTINGS_KEY, folder)
    matplotlib.rcParams["savefig.directory"] = folder


def _start_path(directory):
    """Make the path where a dialog starts. An empty text or a bare file name goes into the
    last folder. A path that has a folder in it is used as it is.

    Args:
        directory (str): Empty text, a file name, or a path.

    Returns:
        path (str): The path to give to the dialog.
    """
    if directory and os.path.dirname(directory):
        return directory
    folder = last_directory()
    return os.path.join(folder, directory) if folder else directory


def open_file_name(parent, caption, directory="", filter=""):
    """Ask the user for one file to open. The dialog starts in the folder used last.

    Args:
        parent (QWidget): Parent of the dialog.
        caption (str): Title of the dialog.
        directory (str): Empty text, or a file name or path to suggest.
        filter (str): File filter, for example "FITS files (*.fits)".

    Returns:
        result (tuple): (path, selected_filter). The path is an empty text if the user cancels.
    """
    path, selected = QFileDialog.getOpenFileName(parent, caption, _start_path(directory), filter)
    if path:
        _remember(path)
    return path, selected


def open_file_names(parent, caption, directory="", filter=""):
    """Ask the user for one or more files to open. The dialog starts in the folder used last.

    Args:
        parent (QWidget): Parent of the dialog.
        caption (str): Title of the dialog.
        directory (str): Empty text, or a file name or path to suggest.
        filter (str): File filter, for example "FITS files (*.fits)".

    Returns:
        result (tuple): (paths, selected_filter). The list is empty if the user cancels.
    """
    paths, selected = QFileDialog.getOpenFileNames(parent, caption, _start_path(directory), filter)
    if paths:
        _remember(paths[0])
    return paths, selected


def save_file_name(parent, caption, directory="", filter=""):
    """Ask the user for the path of a file to save. The dialog starts in the folder used last.

    Args:
        parent (QWidget): Parent of the dialog.
        caption (str): Title of the dialog.
        directory (str): Empty text, or a file name or path to suggest.
        filter (str): File filter, for example "FITS files (*.fits)".

    Returns:
        result (tuple): (path, selected_filter). The path is an empty text if the user cancels.
    """
    path, selected = QFileDialog.getSaveFileName(parent, caption, _start_path(directory), filter)
    if path:
        _remember(path)
    return path, selected


def use_last_directory_in_plot_toolbars():
    """Make the save button of the plot toolbars start in the folder that the user used last.
    Call it once at start-up. A plot toolbar that saves a file does not change the stored folder.

    Args:
        None

    Returns:
        None
    """
    folder = last_directory()
    if folder:
        matplotlib.rcParams["savefig.directory"] = folder
