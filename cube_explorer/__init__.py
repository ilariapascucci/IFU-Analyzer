"""
cube_explorer package
Contains modules for IFU data cube visualization and analysis
"""
import contextlib
import io

# HAPI prints a licence banner when it is imported. Import it once here with the output
# hidden, so the banner stays out of the console and out of every worker process.
with contextlib.redirect_stdout(io.StringIO()):
    try:
        import hapi  # noqa: F401
    except ImportError:
        pass
