==================
Keyboard Shortcuts
==================

Complete reference for keyboard shortcuts in IFU Analyzer.

File Operations
===============

==================== ==============================
Shortcut             Action
==================== ==============================
Ctrl+O               Open FITS file
Ctrl+S               Save project
Ctrl+E               Export spectrum
Ctrl+Q               Quit application
Ctrl+Shift+R         Export region
==================== ==============================

Navigation & Viewing
====================

==================== ==============================
Shortcut             Action
==================== ==============================
Ctrl++               Zoom in
Ctrl+-               Zoom out
Ctrl+0               Reset zoom (fit to window)
Left/Right Arrow     Previous/next wavelength
Up/Down Arrow        Increase/decrease wavelength step
Spacebar+Drag        Pan view (when zoomed)
Home                 First wavelength
End                  Last wavelength
Page Up/Down          Jump 10 wavelengths
==================== ==============================

Region Selection
================

==================== ==============================
Shortcut             Action
==================== ==============================
R                    Activate region selector
R+1                  Point selection tool
R+2                  Rectangle selection tool
R+3                  Circle selection tool
R+4                  Ellipse selection tool
R+5                  Polygon selection tool
Shift+Drag           Add to selection
Ctrl+Drag            Subtract from selection
Alt+Drag             Keep only intersection
U (when in polygon)   Undo last polygon point
Delete               Clear current selection
Escape               Cancel operation
==================== ==============================

Analysis & Fitting
==================

==================== ==============================
Shortcut             Action
==================== ==============================
F                    Fit single pixel (interactive)
C                    Fit full cube (batch)
Ctrl+F               Fit options dialog
Shift+F              View fit results
==================== ==============================

Display Options
===============

==================== ==============================
Shortcut             Action
==================== ==============================
L                    Linear color scale
M                    Logarithmic color scale
P                    Power law color scale
S                    Square root color scale
G                    Toggle grid overlay
Ctrl+G               Configure grid
B                    Show/hide colorbar
I                    Invert colormap
J                    Next colormap
K                    Previous colormap
H                    Histogram adjustment
Ctrl+L               Color scale dialog
==================== ==============================

Spectrum Operations
===================

==================== ==============================
Shortcut             Action
==================== ==============================
Ctrl+Shift+E         Export spectrum (advanced)
Ctrl+X               Cut spectrum
Ctrl+C               Copy spectrum
Ctrl+V               Paste spectrum
Right-click          Spectrum context menu
==================== ==============================

Tools & Features
================

==================== ==============================
Shortcut             Action
==================== ==============================
T                    Show/hide toolbar
Ctrl+,               Preferences dialog
Ctrl+;               View settings
Ctrl+.               Tool options
==================== ==============================

Help & Documentation
====================

==================== ==============================
Shortcut             Action
==================== ==============================
F1                   Open documentation
Shift+F1             Keyboard shortcuts
?                    Quick help
H                    Show this help
Ctrl+H               Full manual
==================== ==============================

Context Menu
============

Right-click provides context-sensitive menus:

**On Intensity Map:**

- Rotate view
- Mirror image
- Statistics
- Clear selection
- Add measurement

**On Spectrum:**

- Fit Gaussian
- Set baseline
- Export spectrum
- Zoom spectral region

**In Empty Area:**

- Load region
- Save region
- Clear all

Special Keys
============

==================== ==============================
Key                  Function
==================== ==============================
Enter                Confirm selection, apply changes
Escape               Cancel operation, deselect
Tab                  Move between interface panels
Shift+Tab            Move to previous panel
Mouse Scroll         Zoom in/out
Middle Mouse+Drag    Pan view
==================== ==============================

Platform-Specific Differences
=============================

**Windows/Linux**

- Alt for menu access
- Ctrl+Z for undo
- Ctrl+Y for redo

**macOS**

- Cmd replaces Ctrl (e.g., Cmd+O for Open)
- Option for Alt key
- ⌘+Z for undo
- ⌘+⇧+Z for redo

Customizing Shortcuts
=====================

To customize keyboard shortcuts:

1. Tools → Preferences → Keyboard
2. Select command to modify
3. Enter new shortcut
4. Click OK
5. Restart application for some changes

**Note**: Some shortcuts are hard-coded and cannot be modified.

Commonly Used Shortcut Combinations
===================================

**Typical Workflow**

1. Ctrl+O → Open FITS file
2. R → Select region
3. F → Fit single pixel
4. C → Fit full cube
5. Ctrl+E → Export spectrum
6. Ctrl+S → Save results

**Quick Navigation**

1. Left/Right arrows → Move through wavelengths
2. Ctrl+0 → Reset view
3. R → Select region
4. View results

**Analysis Session**

1. Ctrl+O → Open file
2. L/M/P/S → Change color scale
3. G → Add grid
4. F or C → Perform fitting
5. Ctrl+Shift+E → Export

**Exporting Multiple**

1. Select region
2. Ctrl+E → Export
3. Right arrow → Move to next wavelength
4. Repeat Ctrl+E as needed

Accessibility
==============

**Navigation Without Mouse**

- Tab/Shift+Tab: Move between controls
- Arrow keys: Select options
- Enter: Activate buttons
- Escape: Cancel dialogs

**High Contrast Mode**

- View → Display Options → High Contrast
- Improves visibility on certain displays

**Font Size**

- Tools → Preferences → Font Size
- Increase text size for visibility

Tips for Efficiency
===================

1. **Memorize most-used shortcuts**: Ctrl+O, Ctrl+E, R, F, C
2. **Use number keys** for region tool selection (R+1, R+2, etc.)
3. **Arrow keys** faster than slider for wavelength navigation
4. **Right-click** for context-sensitive options
5. **F1** when in doubt about functionality

Troubleshooting Shortcuts
==========================

**Shortcut not working**

- Check for conflicting system shortcuts (Windows/macOS)
- Ensure application window is active (has focus)
- Try menu equivalent instead
- Restart application if recently customized

**Modifier key issues**

- Verify Caps Lock is not on
- Try holding modifier longer
- Check keyboard layout (affects some combinations)

**Custom shortcuts not saving**

- Ensure you clicked OK/Apply in preferences
- Restart application to apply changes
- Check permissions on configuration file

Quick Reference Card
====================

Print this quick reference:

.. code-block:: text

   ┌─────────────────────────────────────┐
   │ IFU ANALYZER QUICK SHORTCUTS        │
   ├─────────────────────────────────────┤
   │ File:                               │
   │   Ctrl+O  Open    Ctrl+S  Save     │
   │                                     │
   │ Select:                             │
   │   R  Activate    R+1-5  Choose     │
   │   Shift+Drag  Add  Ctrl+Drag  Sub  │
   │                                     │
   │ Analysis:                           │
   │   F  Fit Single    C  Fit Cube     │
   │                                     │
   │ Display:                            │
   │   L  Linear  M  Log  P  Power      │
   │   Ctrl++ Zoom  Ctrl+0 Reset        │
   │                                     │
   │ Help:                               │
   │   F1  Manual    ?  Help             │
   └─────────────────────────────────────┘

Next Steps
==========

- Memorize key shortcuts for efficiency
- Discover additional shortcuts through menus
- Customize as needed for your workflow
- See :doc:`command_reference` for menu commands
